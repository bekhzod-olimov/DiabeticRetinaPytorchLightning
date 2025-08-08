import os, cv2
from tqdm import tqdm
 
def rotate_180_and_save(root_dir, extensions=(".jpg", ".png", ".jpeg", ".tif", ".bmp")):
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in tqdm(filenames, desc="Processing..."):
            if fname.lower().endswith(extensions) and "_180" not in fname:
                full_path = os.path.join(dirpath, fname)
                # 이미지 읽기
                img = cv2.imread(full_path)
                if img is None:
                    print(f"이미지 로드 실패: {full_path}")
                    continue
                # 180도 회전
                rotated = cv2.rotate(img, cv2.ROTATE_180)
 
                # 저장 경로 설정
                name, ext = os.path.splitext(fname)
                new_name = f"{name}_180{ext}"
                save_path = os.path.join(dirpath, new_name)
 
                # 이미지 저장
                cv2.imwrite(save_path, rotated)
                # print(f"저장 완료: {save_path}")
 
rotate_180_and_save("/home/super/Desktop/bekhzod/backup/tile_classification/25_05_09")