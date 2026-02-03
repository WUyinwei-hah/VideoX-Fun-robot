import argparse
import json
import os
import random
import re
from typing import Dict, List, Optional, Tuple

from decord import VideoReader


SEP = "ad23r2 the screen suddenly turns white, then the camera view suddenly changes."


def _sanitize_character_component(name: str) -> str:
    n = str(name).strip()
    n = re.sub(r"\s+", "_", n)
    n = re.sub(r"[^A-Za-z0-9._-]", "_", n)
    n = re.sub(r"_+", "_", n)
    n = n.strip("._-")
    return n


def _guess_primary_name(character_key: str, names: List[str]) -> Optional[str]:
    folder = os.path.basename(character_key)
    folder_norm = _sanitize_character_component(folder)
    for nm in names:
        if _sanitize_character_component(nm) == folder_norm:
            return nm
    return None


def _build_extra_id_map(character_key: str, src_names: List[str], edt_names: List[str]) -> Dict[str, str]:
    all_names: List[str] = []
    seen = set()
    for nm in (src_names or []) + (edt_names or []):
        nm = str(nm).strip()
        if not nm or nm in seen:
            continue
        seen.add(nm)
        all_names.append(nm)

    primary = _guess_primary_name(character_key, all_names)

    ordered: List[str] = []
    if primary is not None:
        ordered.append(primary)
    for nm in sorted(all_names):
        if nm == primary:
            continue
        ordered.append(nm)

    mapping: Dict[str, str] = {}
    for i, nm in enumerate(ordered):
        mapping[nm] = f"<extra_id_{i}>"
    return mapping


def _anonymize_prompt(text: str, mapping: Dict[str, str]) -> str:
    if not text or not mapping:
        return text
    out = str(text)
    for nm in sorted(mapping.keys(), key=len, reverse=True):
        out = re.sub(re.escape(nm), mapping[nm], out)
    return out


def _find_video_for_json(video_root: str, rel_dir: str, stem: str, exts: Tuple[str, ...]) -> Optional[str]:
    cand_dir = os.path.join(video_root, rel_dir)
    for ext in exts:
        cand = os.path.join(cand_dir, stem + ext)
        if os.path.exists(cand):
            return cand
    return None


def _scene_id_from_stem(stem: str) -> str:
    return re.sub(r"__\d+$", "", str(stem))


def _safe_video_len(path: str) -> Optional[int]:
    try:
        vr = VideoReader(path)
        n = len(vr)
        del vr
        return int(n)
    except Exception:
        return None


def _build_index(
    recaption_root: str,
    video_root: str,
    recaption_key: str,
    video_exts: Tuple[str, ...],
):
    root = os.path.abspath(recaption_root)
    video_root = os.path.abspath(video_root)

    character_to_clips = {}

    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if not fn.lower().endswith(".json"):
                continue
            if fn == "report.json":
                continue

            json_path = os.path.join(dirpath, fn)
            rel = os.path.relpath(json_path, root)
            rel_dir = os.path.dirname(rel)

            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                continue

            prompt = data.get(recaption_key, "")
            if not isinstance(prompt, str):
                prompt = str(prompt)
            prompt = prompt.strip()
            if not prompt:
                continue

            characters = data.get("Characters", {})
            if isinstance(characters, dict):
                character_names = [str(k).strip() for k in characters.keys() if str(k).strip()]
            else:
                character_names = []

            stem = os.path.splitext(fn)[0]
            video_path = _find_video_for_json(video_root, rel_dir, stem, video_exts)
            if video_path is None:
                continue

            scene_id = _scene_id_from_stem(stem)

            character_to_clips.setdefault(rel_dir, []).append(
                {
                    "video_path": video_path,
                    "prompt": prompt,
                    "json_path": json_path,
                    "character_names": character_names,
                    "scene_id": scene_id,
                }
            )

    characters = []
    for k, v in character_to_clips.items():
        if not isinstance(v, list) or len(v) < 2:
            continue
        scene_ids = {c.get('scene_id') for c in v if isinstance(c, dict)}
        scene_ids.discard(None)
        if len(scene_ids) < 2:
            continue
        characters.append(k)
    characters.sort()
    if not characters:
        raise ValueError(
            f"No characters with >=2 clips found. recaption_root={recaption_root}, video_root={video_root}"
        )

    return characters, character_to_clips


def _sample_pair_for_character(
    rng: random.Random,
    character_key: str,
    clips: List[dict],
    pairing_strategy: str,
    max_pair_candidates: int,
):
    if len(clips) < 2:
        raise ValueError(f"Character has <2 clips: {character_key}")

    for _ in range(16):
        a_idx = rng.randrange(len(clips))
        a = clips[a_idx]
        a_scene = a.get('scene_id', None) if isinstance(a, dict) else None

        others = [c for i, c in enumerate(clips) if i != a_idx]
        if a_scene is not None:
            others = [c for c in others if c.get('scene_id', None) != a_scene]
        if not others:
            continue

        if str(pairing_strategy).lower() == "closest_length":
            a_len = _safe_video_len(a["video_path"])
            candidates = others
            if max_pair_candidates > 0 and len(candidates) > max_pair_candidates:
                candidates = rng.sample(candidates, max_pair_candidates)

            best = None
            best_diff = None
            for c in candidates:
                c_len = _safe_video_len(c["video_path"])
                if a_len is None or c_len is None:
                    continue
                diff = abs(int(c_len) - int(a_len))
                if best is None or diff < best_diff:
                    best = c
                    best_diff = diff

            if best is not None:
                return a, best

        return a, rng.choice(others)

    a_idx = rng.randrange(len(clips))
    a = clips[a_idx]
    others = [c for i, c in enumerate(clips) if i != a_idx]
    if not others:
        raise ValueError(f"Character has no other clip: {character_key}")
    return a, rng.choice(others)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--recaption_root", type=str, required=True)
    parser.add_argument("--video_root", type=str, default=None)
    parser.add_argument("--recaption_key", type=str, default="Recaption Prompt")
    parser.add_argument("--output_jsonl", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pairing_strategy", type=str, default="random", choices=("random", "closest_length"))
    parser.add_argument("--max_pair_candidates", type=int, default=64)
    args = parser.parse_args()

    video_root = args.video_root or args.recaption_root

    characters, character_to_clips = _build_index(
        recaption_root=args.recaption_root,
        video_root=video_root,
        recaption_key=args.recaption_key,
        video_exts=(".avi", ".mp4", ".mkv", ".webm", ".mov"),
    )

    rng = random.Random(int(args.seed))
    if int(args.num_samples) > len(characters):
        raise ValueError(f"num_samples={args.num_samples} > available_characters={len(characters)}")

    chosen = rng.sample(characters, int(args.num_samples))

    os.makedirs(os.path.dirname(os.path.abspath(args.output_jsonl)), exist_ok=True)
    with open(args.output_jsonl, "w", encoding="utf-8") as wf:
        for i, character_key in enumerate(chosen):
            clips = character_to_clips[character_key]
            src, edt = _sample_pair_for_character(
                rng=rng,
                character_key=character_key,
                clips=clips,
                pairing_strategy=args.pairing_strategy,
                max_pair_candidates=int(args.max_pair_candidates),
            )

            name_map = _build_extra_id_map(
                character_key=character_key,
                src_names=src.get("character_names", []) or [],
                edt_names=edt.get("character_names", []) or [],
            )
            src_prompt = _anonymize_prompt(src.get("prompt", ""), name_map).strip()
            edt_prompt = _anonymize_prompt(edt.get("prompt", ""), name_map).strip()

            item = {
                "index": i,
                "character_key": character_key,
                "source_abs": src["video_path"],
                "edited_abs": edt["video_path"],
                "source_json": src.get("json_path"),
                "edited_json": edt.get("json_path"),
                "refined_source_prompt": src_prompt,
                "edited_video_prompt": edt_prompt,
                "sep": SEP,
                "prompt": (src_prompt + " " + SEP + " " + edt_prompt).strip(),
                "seed": int(args.seed) + i,
            }
            wf.write(json.dumps(item, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
