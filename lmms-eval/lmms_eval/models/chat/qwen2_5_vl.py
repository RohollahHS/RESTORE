import time
import os
from typing import List

import re
import json
import glob
import numpy as np
import hashlib

import torch
from loguru import logger as eval_logger
from tqdm import tqdm

try:
    import decord
except ImportError:
    decord = None

from lmms_eval import utils
from lmms_eval.api.instance import GenerationResult, Instance, TokenCounts
from lmms_eval.api.registry import register_model
from lmms_eval.imports import optional_import
from lmms_eval.models.model_utils.gen_metrics import log_metrics
from lmms_eval.models.simple.qwen2_5_vl import Qwen2_5_VL as Qwen2_5_VLSimple
from lmms_eval.models.simple.qwen2_5_vl import _extract_task_query
from lmms_eval.protocol import ChatMessages

process_vision_info, _has_qwen_vl = optional_import("qwen_vl_utils", "process_vision_info")
if not _has_qwen_vl:
    eval_logger.warning("Failed to import qwen_vl_utils; Please install it via `pip install qwen-vl-utils`")


def _qwen_prune_verbose() -> bool:
    v = os.environ.get("QWEN_PRUNE_VERBOSE", "0")
    return str(v).strip().lower() in {"1", "true", "t", "yes", "y", "on"}


def load_json_with_fallback(file_path):
    # First: try normal JSON loading
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        print(f"Warning: invalid JSON in {file_path}")
        print(f"JSON error: {e}")
        print("Attempting to recover valid key/value pairs...")
    # Fallback: recover valid "key": value pairs
    data = {}
    # Match lines containing a valid JSON string key followed by a value.
    pattern = re.compile(r'^\s*"((?:\\.|[^"\\])*)"\s*:\s*(.*?)(?:,\s*)?$')
    with open(file_path, "r") as f:
        for line_number, line in enumerate(f, 1):
            match = pattern.match(line)
            if not match:
                continue
            key = match.group(1)
            value_str = match.group(2).strip()
            try:
                # Decode the JSON key properly
                key = json.loads(f'"{key}"')
                # Decode the value
                value = json.loads(value_str)
                data[key] = value
            except json.JSONDecodeError:
                # Invalid key/value -> discard it
                continue
    print(f"Recovered {len(data)} valid key/value pairs.")
    return data

def merge_qa_dicts(qa_dir, QA_ID=None):
    if QA_ID is None:
        QA_ID = str(os.getenv("QA_ID", 0))
    pattern = os.path.join(qa_dir, f"qa_{QA_ID}_*.json")
    merged = {}
    for file_path in glob.glob(pattern):
        print(
            f"{50*'='}\n"
            f"Loading... {file_path}\n"
            f"{50*'='}"
        )
        data = load_json_with_fallback(file_path)
        merged.update(data)
    return merged


def save_json_file(file_path, data):
    with open(file_path, "w") as file:
        json.dump(data, file, indent=4)


def image_to_key(img):
    img_array = np.array(img)
    return hashlib.sha256(img_array.tobytes()).hexdigest()


def format_time(seconds):
    """Convert seconds to HH:MM:SS string."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


@register_model("qwen2_5_vl_chat")
class Qwen2_5_VL(Qwen2_5_VLSimple):
    is_simple = False

    def generate_until(self, requests: List[Instance]) -> List[GenerationResult]:
        qa = {}
        default_save_dir = os.path.join(os.getenv("SAVE_LOGS_DIR"), 'eval_logs', 'question_answer_pairs')
        os.makedirs(default_save_dir, exist_ok=True)
        QA_ID = str(os.getenv('QA_ID', 0))
        RANK = self.RANK
        if not int(os.getenv("LOCAL", 0)):
            qa = merge_qa_dicts(default_save_dir, QA_ID)
            print(f"RANK: {RANK} - reading {len(qa)} qa pairs")
        file_path = os.path.join(default_save_dir, f"qa_{QA_ID}_{RANK}.json")
        print(f"RANK: {RANK} - os.path.exists(file_path): {os.path.exists(file_path)}")

        res = []

        # A dummy collate here to sort by doc id
        def _collate(x):
            return x[0], x[0]

        # we group requests by their generation_kwargs,
        # so that we don't try to execute e.g. greedy sampling and temp=0.8 sampling
        # in the same batch.
        re_ords = utils.Collator(
            [reg.args for reg in requests],
            _collate,
            group_fn=lambda x: x[2],
            grouping=True,
        )
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        num_iters = len(requests) // self.batch_size if len(requests) % self.batch_size == 0 else len(requests) // self.batch_size + 1
        # pbar = tqdm(total=num_iters, disable=(self.rank != 0), desc="Model Responding")
        total_elapsed_time = 0
        total_tokens = 0
        iter = 0
        for chunk in chunks:
            ctx, doc_to_messages, all_gen_kwargs, doc_id, task, split = zip(*chunk)
            chat_messages = [doc_to_messages[idx](self.task_dict[task][split][ids]) for idx, (ids, task, split) in enumerate(zip(doc_id, task, split))]
            chat_messages: List[ChatMessages] = [ChatMessages(**{"messages": message}) for message in chat_messages]
            visuals = []
            videos = []
            for messages in chat_messages:
                visual, video, _ = messages.extract_media()
                visuals.append(visual)
                videos.append(video)
            visuals = self.flatten(visuals)
            videos = self.flatten(videos)
            gen_kwargs = all_gen_kwargs[0]

            img_key = ''
            sample = self.task_dict[task[0]][split[0]][doc_id[0]]
            if 'filename' in sample:
                try:
                    img_key = self.task_dict[task[0]][split[0]][doc_id[0]]['file_name']
                except:
                    img_key = image_to_key(visuals[0])
            elif isinstance(visuals, list) and visuals != []:
                img_key = image_to_key(visuals[0])
            elif isinstance(videos, list) and videos != []:
                img_key = videos[0]
            else:
                img_key = image_to_key(visuals)
            request_key = ctx[0].replace(" ", "_") + "_" + str(doc_id[0]) + "_" + img_key

            if request_key in qa.keys():
                res.append(qa[request_key])
                iter += 1
                continue

            # Apply chat template
            video_kwargs = {
                "max_pixels": self.max_pixels,
                "min_pixels": self.min_pixels,
            }
            if self.fps is not None:
                video_kwargs["fps"] = self.fps
            else:
                # Probe videos to get frame count and set nframes = min(max_num_frames, total_frames)
                # This avoids the error when video has fewer frames than max_num_frames
                if videos and decord is not None:
                    try:
                        video_path = videos[0]  # Assume batch size 1 for videos
                        vr = decord.VideoReader(video_path)
                        video_total_frames = len(vr)
                        nframes = min(self.max_num_frames, video_total_frames)
                        # qwen_vl_utils requires nframes to be a multiple of 2 (FRAME_FACTOR)
                        # and rounds using round_by_factor, so we need to floor to even number
                        # to avoid rounding up past total_frames
                        nframes = (nframes // 2) * 2  # Floor to nearest even number
                        nframes = max(2, nframes)  # At least 2 frames
                        video_kwargs["nframes"] = nframes
                    except Exception as e:
                        eval_logger.warning(f"Failed to probe video {videos[0]}: {e}, using default nframes")
                        video_kwargs["nframes"] = self.max_num_frames
                else:
                    video_kwargs["nframes"] = self.max_num_frames
            batched_messages = [chat_message.to_hf_messages(video_kwargs=video_kwargs) for chat_message in chat_messages]
            # Enforce fixed image resolution for Qwen image path as well (not only video kwargs).
            for conv in batched_messages:  # conv: List[Dict]
                for turn in conv:
                    if not isinstance(turn, dict):
                        continue
                    for c in turn.get("content", []):
                        if isinstance(c, dict) and c.get("type") == "image":
                            c["resized_height"] = self.resized_height
                            c["resized_width"] = self.resized_width
                            c["max_pixels"] = self.max_pixels
                            c["min_pixels"] = self.min_pixels
            texts = self.processor.apply_chat_template(batched_messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs, video_kwargs_qwen = process_vision_info(
                batched_messages,
                return_video_kwargs=True,
                return_video_metadata=True,
            )

            video_metadatas = None
            if video_inputs is not None:
                video_inputs, video_metadatas = zip(*video_inputs)
                video_inputs, video_metadatas = list(video_inputs), list(video_metadatas)

            padding_side = "left" if self.batch_size > 1 else "right"
            inputs = self.processor(
                text=texts,
                images=image_inputs,
                videos=video_inputs,
                video_metadata=video_metadatas,
                padding=True,
                padding_side=padding_side,
                return_tensors="pt",
            )

            if self.device_map == "auto":
                inputs = inputs.to("cuda")
            else:
                inputs = inputs.to(self.device)

            if bool(getattr(self._config, "qwen_prune_enable", False)):
                qtxt = _extract_task_query(
                    contexts=list(ctx),
                    batched_messages=batched_messages,
                    task_dict=self.task_dict,
                    task=task[0] if len(task) > 0 else None,
                    split=split[0] if len(split) > 0 else None,
                    doc_id=doc_id,
                )
                self._config.qwen_prune_query_text = qtxt
                self.model.config.qwen_prune_query_text = qtxt
                with torch.no_grad():
                    _ = self.model.get_image_features(
                        pixel_values=inputs["pixel_values"],
                        image_grid_thw=inputs["image_grid_thw"],
                    )
                keep_counts = getattr(self._config, "qwen_prune_last_selected_tokens_batch", None)
                inputs = self._compact_image_placeholders(inputs, keep_counts)
                if keep_counts is not None and self.rank == 0 and _qwen_prune_verbose():
                    print(f"[QWEN_CHAT_PLACEHOLDER_COMPACT] keep_counts={keep_counts}", flush=True)

            # Set default generation kwargs
            default_gen_kwargs = {
                "max_new_tokens": 128,
                "temperature": 0.0,  # Set to 0 for greedy default
                "top_p": None,
                "num_beams": 1,
            }
            # Update with provided kwargs
            current_gen_kwargs = {**default_gen_kwargs, **gen_kwargs}
            pad_token_id = self.tokenizer.pad_token_id

            if current_gen_kwargs["temperature"] > 0:
                current_gen_kwargs["do_sample"] = True
            else:
                current_gen_kwargs["do_sample"] = False
                current_gen_kwargs["temperature"] = None
                current_gen_kwargs["top_p"] = None
                current_gen_kwargs["top_k"] = None

            start_time = time.time()
            cont = self.model.generate(
                **inputs,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=pad_token_id,
                do_sample=current_gen_kwargs["do_sample"],
                temperature=current_gen_kwargs["temperature"],
                top_p=current_gen_kwargs["top_p"],
                num_beams=current_gen_kwargs["num_beams"],
                max_new_tokens=current_gen_kwargs["max_new_tokens"],
                top_k=current_gen_kwargs.get("top_k", None),
                use_cache=self.use_cache,
            )
            end_time = time.time()

            generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, cont)]
            answers = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

            # Calculate timing metrics for batch
            total_elapsed_time += end_time - start_time
            total_tokens += sum(len(ids) for ids in generated_ids_trimmed)

            for i, (ans, context) in enumerate(zip(answers, texts)):
                res.append(GenerationResult(text=ans, token_counts=TokenCounts(output_tokens=len(generated_ids_trimmed[i]))))
                self.cache_hook.add_partial("generate_until", (context, gen_kwargs), ans)

                eval_logger.debug(f"Question: {context}")
                eval_logger.debug(f"Model Response: {ans}")
            # reorder this group of results back to original unsorted form
            qa[request_key] = answers[0]
            save_json_file(file_path, qa)

            if iter % 200 == 0:
                print(f"RANK: {RANK} - [{iter}/{num_iters}] Model Responding..., output: {answers[0]}")
            iter += 1

        res = re_ords.get_original(res)

        # Calculate average speed
        avg_speed = total_tokens / total_elapsed_time if total_elapsed_time > 0 else 0
        # Log metrics
        metric_dict = {
            "total_gen_tokens": total_tokens,
            "total_elapsed_time": total_elapsed_time,
            "avg_speed": avg_speed,
            "additional_metrics": {
                "rank": self.rank,
            },
        }
        log_metrics(**metric_dict)

        return res
