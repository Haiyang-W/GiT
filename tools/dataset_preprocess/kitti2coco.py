import os
import json

def convert_kitti_to_coco(kitti_labels_dir, output_dir, image_files, data_type):
    coco_data = {
        "info": {},
        "licenses": [],
        "categories": [
            {"id": 1, "name": "Car"},
            {"id": 2, "name": "Van"},
            {"id": 3, "name": "Truck"},
            {"id": 4, "name": "Pedestrian"},
            {"id": 5, "name": "Person_sitting"},
            {"id": 6, "name": "Cyclist"},
            {"id": 7, "name": "Tram"},
            {"id": 8, "name": "Misc"}
        ],
        "images": [],
        "annotations": []
    }

    # Read and process each KITTI label file for the given image files
    for image_file_name in image_files:
        file_num = int(image_file_name.split('.')[0])

        with open(os.path.join(kitti_labels_dir, f"{file_num:06d}.txt"), "r") as file:
            lines = file.readlines()

            image_data = {
                "id": file_num,
                "width": 1280,  # 1382 / 1280
                "height": 384,  # 512 / 384
                "file_name": image_file_name,
            }

            coco_data["images"].append(image_data)

            for line in lines:
                elements = line.strip().split()
                class_name = elements[0]
                truncation = float(elements[1])
                occlusion = int(elements[2])
                alpha = float(elements[3])
                bbox = [float(coord) for coord in elements[4:8]]
                dimensions = [float(dim) for dim in elements[8:11]]
                location = [float(coord) for coord in elements[11:14]]
                rotation_y = float(elements[14])

                category_id = None
                for category in coco_data["categories"]:
                    if category["name"] == class_name:
                        category_id = category["id"]
                        break

                if category_id is not None:
                    annotation = {
                        "id": len(coco_data["annotations"]) + 1,
                        "image_id": file_num,
                        "category_id": category_id,
                        "bbox": [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]],
                        "area": (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]),
                        "iscrowd": 0,
                        "truncated": truncation,
                        "occluded": occlusion,
                        "alpha": alpha,
                        "dimensions": dimensions,
                        "location": location,
                        "rotation_y": rotation_y
                    }

                    coco_data["annotations"].append(annotation)

    # Save the coco_data to a JSON file in the specified output directory
    file_name = f"kitti_coco_format_{data_type}.json"
    output_file = os.path.join(output_dir, file_name)
    with open(output_file, "w") as json_file:
        json.dump(coco_data, json_file)

if __name__ == "__main__":
    dir = "data/"
    train_image_path = dir + "train/image/"
    train_label_path = dir + "train/label/"
    val_image_path = dir + "val/image/"
    val_label_path = dir + "val/label/"
    train_coco_path = dir + "train/coco/"
    val_coco_path = dir + "val/coco/"

    # Create directories for train and val if they don't exist
    os.makedirs(train_coco_path, exist_ok=True)
    os.makedirs(val_coco_path, exist_ok=True)

    # Get the list of image files for train and val sets
    train_image_files = os.listdir(train_image_path)
    val_image_files = os.listdir(val_image_path)

    # Convert train set to COCO format and save as JSON
    convert_kitti_to_coco(train_label_path, train_coco_path, train_image_files, "train")

    # Convert val set to COCO format and save as JSON
    convert_kitti_to_coco(val_label_path, val_coco_path, val_image_files, "val")
