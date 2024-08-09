from PIL import Image
import os


def read_bbox(name, data_dir):
    text_bbox_file = os.path.join(data_dir, "text_bounding_box", name + ".txt")
    lead_bbox_file = os.path.join(data_dir, "lead_bounding_box", name + ".txt")

    def convert_to_float(x):
        try:
            return float(x)
        except ValueError:
            return x.strip()

    bbox = {}
    try:
        with open(text_bbox_file, "r") as file1, open(lead_bbox_file, "r") as file2:
            for line1, line2 in zip(file1, file2):
                x_min_text, y_min_text, x_max_text, y_max_text, lead_name = map(
                    convert_to_float, line1.split(",")
                )
                x_min_lead, y_min_lead, x_max_lead, y_max_lead, class_id = map(
                    convert_to_float, line2.split(",")
                )

                if class_id == 1:
                    lead_name += "_long"
                bbox[lead_name] = {
                    "lead": (x_min_lead, y_min_lead, x_max_lead, y_max_lead),
                    "name": (x_min_text, y_min_text, x_max_text, y_max_text),
                }
    except FileNotFoundError as e:
        print(f"File not found: {e}")
    except ValueError as e:
        print(f"Error processing file: {e}")

    return bbox


def crop_bbox(image, bbox):
    def mpl_to_pil_coords(x_mpl, y_mpl):
        x_pil = x_mpl
        y_pil = fig_height - y_mpl  # Inverting the Y-coordinate
        return int(x_pil), int(y_pil)

    fig_width, fig_height = image.width, image.height
    x_min, y_min, x_max, y_max = bbox

    # Convert Matplotlib coordinates to PIL coordinates
    x_min, y_min = mpl_to_pil_coords(x_min, y_min)
    x_max, y_max = mpl_to_pil_coords(x_max, y_max)

    # Validate and adjust coordinates
    x_min, x_max = sorted([x_min, x_max])
    y_min, y_max = sorted([y_min, y_max])

    # Ensure coordinates are within image bounds
    x_min, y_min = max(x_min, 0), max(y_min, 0)
    x_max, y_max = min(x_max, image.width), min(y_max, image.height)

    # Crop the image
    cropped_image = image.crop((x_min, y_min, x_max, y_max))
    return cropped_image


if __name__ == "__main__":
    from PIL import Image

    image = Image.open(
        "/home/fdias/data/CINC_CHALLENGE_2024/GENERATED_IMAGE/PTB-XL/SAMPLE/clean/00015_lr-0.png"
    ).convert("RGB")
    bb = read_bbox(
        "00015_lr-0",
        "/home/fdias/data/CINC_CHALLENGE_2024/GENERATED_IMAGE/PTB-XL/SAMPLE/clean",
    )
    cropped = crop_bbox(image, bb["I"]["lead"])
    cropped
