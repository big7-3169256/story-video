"""Turns story.json into final.mp4 (1080x1920) using only free tools.

story.json shape:
{
  "title": "...", "character": "...", "style": "...", "voice": "en-US-ChristopherNeural" (optional),
  "scenes": [{"text": "narration", "image_prompt": "what is visible"}]
}
"""
import asyncio
import base64
import json
import os
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request

import cv2
import edge_tts
import numpy as np
from PIL import Image

W, H, FPS = 1080, 1920, 30
PAD = 0.3  # seconds of silence after each scene
WORK = "work"
os.makedirs(WORK, exist_ok=True)

story = json.load(open("story.json", encoding="utf-8"))
VOICE = story.get("voice") or "en-US-ChristopherNeural"
STYLE = story.get("style") or "flat 2D cartoon illustration, bold outlines, vibrant colors"
CHARACTER = story.get("character") or ""

# Free image provider: Cloudflare Workers AI (FLUX schnell). Set as GitHub repo secrets.
CF_ACCOUNT = os.environ.get("CF_ACCOUNT_ID", "").strip()
CF_TOKEN = os.environ.get("CF_API_TOKEN", "").strip()
# Camera-parallax needs a depth model. Set PARALLAX=0 in the workflow to use the simple zoom instead.
USE_PARALLAX = os.environ.get("PARALLAX", "1") == "1"


def run(cmd):
    print(" ".join(cmd)[:200])
    subprocess.run(cmd, check=True)


def normalize_image(data, path):
    """Decode any image bytes, crop to 9:16 (centered) and save a clean 1080x1920 PNG."""
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("could not decode image data")
    h, w = img.shape[:2]
    target = W / H
    if w / h > target:  # too wide: crop the sides
        nw = int(h * target)
        x0 = (w - nw) // 2
        img = img[:, x0:x0 + nw]
    else:               # too tall: crop top and bottom
        nh = int(w / target)
        y0 = (h - nh) // 2
        img = img[y0:y0 + nh, :]
    img = cv2.resize(img, (W, H), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(path, img)


def cloudflare_image(prompt):
    url = (
        f"https://api.cloudflare.com/client/v4/accounts/{CF_ACCOUNT}"
        "/ai/run/@cf/black-forest-labs/flux-1-schnell"
    )
    headers = {
        "Authorization": f"Bearer {CF_TOKEN}",
        "Content-Type": "application/json",
        "User-Agent": "Mozilla/5.0",
    }
    # first try a tall image; if the model rejects the size fields, retry with the basic payload
    payloads = [
        {"prompt": prompt[:2000], "steps": 6, "width": 768, "height": 1344},
        {"prompt": prompt[:2000], "steps": 6},
    ]
    last = None
    for payload in payloads:
        for attempt in range(3):
            try:
                req = urllib.request.Request(
                    url, data=json.dumps(payload).encode(), headers=headers, method="POST"
                )
                resp = urllib.request.urlopen(req, timeout=120)
                raw = resp.read()
                if (resp.headers.get("Content-Type") or "").startswith("image/"):
                    return raw
                body = json.loads(raw)
                return base64.b64decode(body["result"]["image"])
            except urllib.error.HTTPError as e:
                last = f"HTTP {e.code}: {e.read()[:300]!r}"
                print("cloudflare:", last)
                if e.code in (400, 422):
                    break  # bad input, go to the simpler payload
                time.sleep(6 * (attempt + 1))
            except Exception as e:  # noqa: BLE001
                last = str(e)
                print("cloudflare:", last)
                time.sleep(6 * (attempt + 1))
    raise RuntimeError(f"Cloudflare image failed: {last}")


def fetch_image(prompt, seed, path):
    full = (
        f"{STYLE}, {CHARACTER}, {prompt}, subject centered in frame, "
        "vertical composition, no text, no watermark"
    )
    if CF_ACCOUNT and CF_TOKEN:
        try:
            normalize_image(cloudflare_image(full), path)
            return
        except Exception as e:  # noqa: BLE001
            print("Cloudflare failed, trying Pollinations:", e)

    # fallback: Pollinations anonymous endpoint (slow, rate limited, may return 402)
    url = (
        "https://image.pollinations.ai/prompt/"
        + urllib.parse.quote(full)
        + f"?width={W}&height={H}&seed={seed}&nologo=true&model=flux"
    )
    for attempt in range(4):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            data = urllib.request.urlopen(req, timeout=180).read()
            if len(data) > 10000:
                normalize_image(data, path)
                return
        except Exception as e:  # noqa: BLE001
            print("pollinations retry:", e)
        time.sleep(16)
    raise RuntimeError(f"Image generation failed for: {prompt[:60]}")


async def tts(text, mp3_path):
    comm = edge_tts.Communicate(text, VOICE, rate="+8%", boundary="WordBoundary")
    words = []
    with open(mp3_path, "wb") as f:
        async for chunk in comm.stream():
            if chunk["type"] == "audio":
                f.write(chunk["data"])
            elif chunk["type"] == "WordBoundary":
                start = chunk["offset"] / 1e7
                end = (chunk["offset"] + chunk["duration"]) / 1e7
                words.append((start, end, chunk["text"]))
    return words


def audio_duration(path):
    out = subprocess.check_output(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", path]
    )
    return float(out.strip())


def ass_time(t):
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = t % 60
    return f"{h}:{m:02d}:{s:05.2f}"


def build_ass(all_words, path):
    header = (
        "[Script Info]\nScriptType: v4.00+\nPlayResX: %d\nPlayResY: %d\n\n"
        "[V4+ Styles]\n"
        "Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,BackColour,"
        "Bold,Italic,Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,"
        "Alignment,MarginL,MarginR,MarginV,Encoding\n"
        "Style: Default,DejaVu Sans,88,&H00FFFFFF,&H000000FF,&H00000000,&H64000000,"
        "-1,0,0,0,100,100,0,0,1,7,2,2,60,60,420,1\n\n"
        "[Events]\nFormat: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text\n"
    ) % (W, H)
    lines = []
    chunk_size = 3
    for i in range(0, len(all_words), chunk_size):
        group = all_words[i:i + chunk_size]
        start = group[0][0]
        end = group[-1][1]
        # keep the caption up until the next chunk starts so there is no flicker
        if i + chunk_size < len(all_words):
            end = max(end, all_words[i + chunk_size][0])
        text = " ".join(w[2] for w in group).upper().replace("\n", " ")
        lines.append(f"Dialogue: 0,{ass_time(start)},{ass_time(end)},Default,,0,0,0,,{text}")
    with open(path, "w", encoding="utf-8") as f:
        f.write(header + "\n".join(lines) + "\n")


def make_clip(img, mp3, dur, idx, out):
    frames = int(dur * FPS) + 1
    if idx % 2 == 0:
        z = f"1+0.22*on/{frames}"        # slow zoom in
    else:
        z = f"1.22-0.22*on/{frames}"     # slow zoom out
    vf = (
        f"scale={W * 2}:{H * 2},"
        f"zoompan=z='{z}':d={frames}:x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':s={W}x{H}:fps={FPS},"
        "format=yuv420p"
    )
    run([
        "ffmpeg", "-y", "-loglevel", "error", "-i", img, "-i", mp3,
        "-vf", vf, "-af", f"apad=pad_dur={PAD}", "-t", f"{dur:.3f}",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        "-c:a", "aac", "-b:a", "160k", "-ar", "44100", "-ac", "2", out,
    ])


DEPTH = None


def load_depth():
    """Load a small free depth model (CPU). If it fails we fall back to simple zoom."""
    global DEPTH
    try:
        from transformers import pipeline

        DEPTH = pipeline(
            "depth-estimation",
            model="depth-anything/Depth-Anything-V2-Small-hf",
            device=-1,
        )
        print("depth model loaded")
    except Exception as e:  # noqa: BLE001
        DEPTH = None
        print("depth model unavailable, using simple zoom:", e)


def get_depth(img_path):
    """Returns a float32 HxW map in 0..1 where 1 = near the camera."""
    pil = Image.open(img_path).convert("RGB")
    d = np.array(DEPTH(pil)["depth"]).astype("float32")
    d = cv2.resize(d, (W, H), interpolation=cv2.INTER_CUBIC)
    d = (d - d.min()) / (d.max() - d.min() + 1e-6)
    return cv2.GaussianBlur(d, (0, 0), 9)  # soften to avoid tearing at edges


# camera travel direction per scene (x, y) so consecutive scenes feel different
MOTIONS = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 0.5), (-1, -0.5)]


def make_clip_parallax(img_path, mp3, dur, idx, out, depth=None):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
    if depth is None:
        depth = get_depth(img_path)
    frames = int(dur * FPS) + 1

    gx, gy = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    cx, cy = W / 2.0, H / 2.0
    dxd, dyd = MOTIONS[idx % len(MOTIONS)]
    zoom_in = idx % 2 == 0
    amp = 56.0  # max parallax shift in pixels (kept inside the zoom margin)
    near = (depth - 0.5).astype(np.float32)

    proc = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error",
         "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
         "-i", mp3, "-af", f"apad=pad_dur={PAD}", "-t", f"{dur:.3f}",
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
         "-c:a", "aac", "-b:a", "160k", "-ar", "44100", "-ac", "2", out],
        stdin=subprocess.PIPE,
    )
    try:
        for f in range(frames):
            p = f / max(frames - 1, 1)
            e = p * p * (3 - 2 * p)          # ease in/out
            s = (e - 0.5) * 2.0              # -1 .. 1 camera position
            z = (1.06 + 0.06 * e) if zoom_in else (1.12 - 0.06 * e)
            map_x = cx + (gx - cx) / z - near * amp * s * dxd
            map_y = cy + (gy - cy) / z - near * amp * s * dyd
            frame = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REPLICATE)
            proc.stdin.write(frame.tobytes())
    finally:
        proc.stdin.close()
        proc.wait()
    if proc.returncode != 0:
        raise RuntimeError("ffmpeg failed in parallax clip")


async def main():
    if USE_PARALLAX:
        load_depth()
    clips, all_words, offset = [], [], 0.0
    for i, scene in enumerate(story["scenes"]):
        img = f"{WORK}/scene{i}.png"
        mp3 = f"{WORK}/scene{i}.mp3"
        clip = f"{WORK}/clip{i}.mp4"

        words = await tts(scene["text"], mp3)
        dur = audio_duration(mp3) + PAD
        fetch_image(scene["image_prompt"], seed=1000 + i, path=img)
        done = False
        if DEPTH is not None:
            try:
                make_clip_parallax(img, mp3, dur, i, clip)
                done = True
            except Exception as e:  # noqa: BLE001
                print("parallax failed, falling back to zoom:", e)
        if not done:
            make_clip(img, mp3, dur, i, clip)

        all_words += [(s + offset, e + offset, t) for s, e, t in words]
        offset += dur
        clips.append(clip)

    with open(f"{WORK}/list.txt", "w") as f:
        for c in clips:
            f.write(f"file '{os.path.basename(c)}'\n")
    run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
         "-i", f"{WORK}/list.txt", "-c", "copy", f"{WORK}/joined.mp4"])

    build_ass(all_words, f"{WORK}/captions.ass")

    # optional background music: drop a royalty-free file at assets/music.mp3
    if os.path.exists("assets/music.mp3"):
        run(["ffmpeg", "-y", "-loglevel", "error", "-i", f"{WORK}/joined.mp4",
             "-stream_loop", "-1", "-i", "assets/music.mp3",
             "-filter_complex",
             f"[0:v]ass={WORK}/captions.ass[v];[1:a]volume=0.07[m];[0:a][m]amix=inputs=2:duration=first[a]",
             "-map", "[v]", "-map", "[a]", "-c:v", "libx264", "-preset", "veryfast",
             "-crf", "20", "-c:a", "aac", "-shortest", "final.mp4"])
    else:
        run(["ffmpeg", "-y", "-loglevel", "error", "-i", f"{WORK}/joined.mp4",
             "-vf", f"ass={WORK}/captions.ass", "-c:v", "libx264", "-preset", "veryfast",
             "-crf", "20", "-c:a", "copy", "final.mp4"])
    print("done: final.mp4", round(offset, 1), "seconds")


asyncio.run(main())
