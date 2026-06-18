import os
from PIL import Image, ImageDraw, ImageFont
import argparse

def make_grid_2x2(images, labels, tile_size=None, pad=10, font=None):
    """将4张图拼成2x2大图，并在左上角写上文字"""
    assert len(images) == 4
    imgs = [Image.open(p).convert("RGB") for p in images]

    # 统一尺寸
    if tile_size is None:
        max_w = max(im.width for im in imgs)
        max_h = max(im.height for im in imgs)
        tile_w, tile_h = max_w, max_h
    else:
        tile_w = tile_h = tile_size

    grid_w = tile_w * 2 + pad
    grid_h = tile_h * 2 + pad
    canvas = Image.new("RGB", (grid_w, grid_h), color=(255, 255, 255))

    positions = [
        (0, 0),
        (tile_w + pad, 0),
        (0, tile_h + pad),
        (tile_w + pad, tile_h + pad)
    ]

    for im, label, pos in zip(imgs, labels, positions):
        im = im.resize((tile_w, tile_h))
        canvas.paste(im, pos)

        # 写文字
        draw = ImageDraw.Draw(canvas)
        text_pos = (pos[0] + 15, pos[1] + 15)
        draw.text(text_pos, label, fill=(255, 0, 0), font=font)

    return canvas

def make_grid_vertical3(images, labels, tile_size=None, pad=10, font=None):
    """将3张图上下拼接，并在每张图左上角写上文字"""
    assert len(images) == 3
    imgs = [Image.open(p).convert("RGB") for p in images]

    # 统一尺寸
    if tile_size is None:
        max_w = max(im.width for im in imgs)
        max_h = max(im.height for im in imgs)
        tile_w, tile_h = max_w, max_h
    else:
        tile_w, tile_h = tile_size

    # 计算画布大小：宽度一致，高度为3张图 + 2个间隔
    grid_w = tile_w
    grid_h = tile_h * 3 + pad * 2
    canvas = Image.new("RGB", (grid_w, grid_h), color=(255, 255, 255))

    positions = [
        (0, 0),
        (0, tile_h + pad),
        (0, 2 * (tile_h + pad))
    ]

    for im, label, pos in zip(imgs, labels, positions):
        im = im.resize((tile_w, tile_h))
        canvas.paste(im, pos)

        # 写文字
        draw = ImageDraw.Draw(canvas)
        text_pos = (pos[0] + 15, pos[1] + 15)
        draw.text(text_pos, label, fill=(255, 0, 0), font=font)

    return canvas

def batch_make_grids(folder_paths, labels, output_dir, max_count=None, font_path=None):
    """批量生成四宫格拼图"""
    os.makedirs(output_dir, exist_ok=True)
    all_imgs = [sorted([os.path.join(f, x) for x in os.listdir(f) if x.lower().endswith(('.jpg','.png'))]) for f in folder_paths]
    min_len = min(len(lst) for lst in all_imgs)
    if max_count:
        min_len = min(min_len, max_count)

    #font = ImageFont.truetype(font_path, 512) if font_path else None
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size=36)
    #font=ImageFont.truetype("arial.ttf",size=36)


    for i in range(min_len):
        img_paths = [lst[i] for lst in all_imgs]
        if len(img_paths) == 4:
            out_img = make_grid_2x2(img_paths, labels, pad=10, font=font)
        elif len(img_paths) == 3:
            out_img = make_grid_vertical3(img_paths, labels, pad=10, font=font)
        else:
            raise ValueError("当前仅支持3图或4图拼接")
        out_path = os.path.join(output_dir, f"grid_{i+1:03d}.jpg")
        out_img.save(out_path)
        print(f"✅ Saved {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folders", nargs="+", required=True, help="输入文件夹路径（支持3个或4个）")
    parser.add_argument("--labels", nargs="+", required=True, help="对应的文字标签（数量需与文件夹一致）")
    parser.add_argument("--output", required=True, help="输出文件夹")
    parser.add_argument("--font", default=None, help="可选字体文件路径（如：simhei.ttf）")
    parser.add_argument("--max", type=int, default=None, help="最多生成多少张")
    args = parser.parse_args()

    batch_make_grids(args.folders, args.labels, args.output, args.max, args.font)
