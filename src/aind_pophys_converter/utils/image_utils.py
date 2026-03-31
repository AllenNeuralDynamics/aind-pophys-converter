"""Image manipulation utilities."""

from PIL import Image, ImageDraw, ImageFont, ImageOps


def add_border_and_label(
    img: Image.Image,
    label: str,
    border: int = 5,
    font_size: int = 40,
) -> Image.Image:
    """Add a border and a text label to an image (top-center, above image).

    Parameters
    ----------
    img : Image.Image
        Input PIL image.
    label : str
        Text label to add at the top center.
    border : int, optional
        Border size in pixels (default is 5).
    font_size : int, optional
        Font size for the label text (default is 40).

    Returns
    -------
    Image.Image
        New image with border and label added.
    """
    bordered = ImageOps.expand(img, border=border, fill="white")
    if bordered.mode != "RGB":
        bordered = bordered.convert("RGB")
    font = ImageFont.load_default(size=font_size)
    tmp_draw = ImageDraw.Draw(bordered)
    bbox = tmp_draw.textbbox((0, 0), label, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    label_area_h = text_h + 24
    canvas = Image.new(
        "RGB",
        (bordered.width, bordered.height + label_area_h),
        color="white",
    )
    canvas.paste(bordered, (0, label_area_h))
    draw = ImageDraw.Draw(canvas)
    x = (canvas.width - text_w) // 2
    y = (label_area_h - text_h) // 2
    draw.text((x, y), label, fill="black", font=font)
    return canvas
