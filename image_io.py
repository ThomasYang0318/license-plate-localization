import multiprocessing


def _read_image_pair_worker(image_path: str):
    import cv2

    color_image = cv2.imread(image_path)
    gray_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if color_image is None or gray_image is None:
        return False, None, None, "decode failed"
    return True, color_image, gray_image, ""


def read_image_pair(image_path, timeout_seconds: float):
    if timeout_seconds <= 0:
        success, color_image, gray_image, message = _read_image_pair_worker(str(image_path))
        return color_image, gray_image, None if success else message

    try:
        context = multiprocessing.get_context("fork")
    except ValueError:
        context = multiprocessing.get_context()

    pool = context.Pool(processes=1)
    result = pool.apply_async(_read_image_pair_worker, (str(image_path),))
    try:
        success, color_image, gray_image, message = result.get(timeout=timeout_seconds)
    except multiprocessing.TimeoutError:
        pool.terminate()
        pool.join()
        return None, None, f"read timeout after {timeout_seconds:g}s"

    pool.close()
    pool.join()
    return color_image, gray_image, None if success else message
