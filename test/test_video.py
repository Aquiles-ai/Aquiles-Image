from openai import OpenAI
import sys
import time

openai = OpenAI(base_url="http://127.0.0.1:5500", api_key="__UNKNOWN__")

POLL_INTERVAL = 2.0
TIMEOUT_SECONDS = 60 * 30
DOWNLOAD_RETRIES = 5
DOWNLOAD_BACKOFF_BASE = 2

IN_PROGRESS_STATES = {
    "queued", "processing", "in_progress", "running", "starting"
}
SUCCESS_STATES = {
    "succeeded", "completed", "ready", "finished", "success"
}
FAILED_STATES = {"failed", "error"}

def pretty_progress_bar(progress, length=30):
    try:
        p = float(progress or 0.0)
    except Exception:
        p = 0.0
    filled = int((p / 100.0) * length)
    return "=" * filled + "-" * (length - filled), p

def poll_until_done(video_id):
    start = time.time()
    bar_length = 30

    video = openai.videos.retrieve(video_id)

    while True:
        status = (getattr(video, "status", "") or "").lower()

        progress = getattr(video, "progress", None)
        bar, p = pretty_progress_bar(progress, bar_length)
        status_text = status.capitalize() if status else "Unknown"
        sys.stdout.write(f"\r{status_text}: [{bar}] {p:.1f}%")
        sys.stdout.flush()

        if status in SUCCESS_STATES:
            sys.stdout.write("\n")
            print("Final status:", status)
            return video
        if status in FAILED_STATES:
            sys.stdout.write("\n")
            msg = getattr(getattr(video, "error", None), "message", "Video generation failed")
            raise RuntimeError(f"Video generation failed: {msg}")

        elapsed = time.time() - start
        if TIMEOUT_SECONDS and elapsed > TIMEOUT_SECONDS:
            sys.stdout.write("\n")
            raise TimeoutError(f"Timed out after {TIMEOUT_SECONDS} seconds while waiting for video generation (last status: {status})")

        time.sleep(POLL_INTERVAL)
        video = openai.videos.retrieve(video_id)

def download_with_retries(video_id, out_path="video.mp4"):
    attempt = 0
    while attempt < DOWNLOAD_RETRIES:
        attempt += 1
        try:
            print(f"Attempting download (try {attempt}/{DOWNLOAD_RETRIES})...")
            content = openai.videos.download_content(video_id, variant="video")
            content.write_to_file("video.mp4")
            print("Wrote", out_path)
            return out_path

        except Exception as e:
            err_text = str(e)
            print(f"Download error: {err_text}")
            if attempt >= DOWNLOAD_RETRIES:
                raise RuntimeError(f"Failed to download after {DOWNLOAD_RETRIES} attempts: {err_text}")
            backoff = DOWNLOAD_BACKOFF_BASE ** attempt
            backoff = min(backoff, 60)
            print(f"Retrying in {backoff} seconds...")
            time.sleep(backoff)

def main():
    try:
        created = openai.videos.create(
            model="wan2.2",
            prompt="A video of a cool cat on a motorcycle in the night",
        )
    except Exception as e:
        print("Error creating video:", e)
        sys.exit(1)

    video_id = getattr(created, "id", None)
    if not video_id:
        print("No video id returned from create call.")
        sys.exit(1)

    print("Video generation started:", video_id)

    try:
        finished_video = poll_until_done(video_id)
    except TimeoutError as te:
        print("Timeout:", te)
        sys.exit(1)
    except RuntimeError as re:
        print(re)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        sys.exit(1)
    except Exception as e:
        print("Unexpected error while waiting for generation:", e)
        sys.exit(1)

    try:
        download_with_retries(video_id, out_path="video.mp4")
    except Exception as e:
        print("Error downloading or writing video content:", e)
        sys.exit(1)

if __name__ == "__main__":
    main()
