from PIL import Image
import numpy as np
import pathlib

class ImagePreprocessor:
    def __init__(self):
        """
        Initializes the ImagePreprocessor.
        Defines common image extensions to be processed.
        """
        self.image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff', '.webp'}
    
    def resize_image(self, img, base_width = 512):
        # Calculate the aspect ratio
        aspect_ratio = img.height / img.width
        new_height = int(base_width * aspect_ratio)

        # Resize the image
        img = img.resize((base_width, new_height), Image.Resampling.LANCZOS)
        return img

    def background_flatten(self, rgba_img: Image.Image, name_stem: str, output_target_dir: pathlib.Path):
        """
        Composites an RGBA image with gray, black, or white backgrounds and saves them.

        Args:
            rgba_img: PIL.Image object in 'RGBA' mode.
            name_stem: The base name of the file (without extension).
            output_target_dir: The specific pathlib.Path directory where flattened images will be saved.
        """
        bg_colors = {
            "black": (0, 0, 0, 255),
            "white": (255, 255, 255, 255),
            "gray": (128, 128, 128, 255)
        }

        for key, bg_color_tuple in bg_colors.items():
            background = Image.new("RGBA", rgba_img.size, bg_color_tuple)
            # Paste the original image onto the background using its alpha channel as the mask
            background.paste(rgba_img, (0, 0), rgba_img)
            # Convert to RGB (dropping the alpha channel)
            out_rgb = background.convert("RGB")
            self.save_flattened_img(output_target_dir, out_rgb, name_stem, key)

    def save_flattened_img(self, output_dir: pathlib.Path, out_rgb_img: Image.Image, name_stem: str, bg_key: str):
        """
        Saves the flattened RGB image to the specified directory.
        The output filename will be in the format: {name_stem}_{bg_key}.jpg.

        Args:
            output_dir: The pathlib.Path directory to save the image.
            out_rgb_img: The PIL.Image object (RGB) to save.
            name_stem: The base name of the original file.
            bg_key: The key indicating the background color used (e.g., "black", "white").
        """
        # Output filename format: {name_stem}_{bg_key}.jpg
        output_filename = f"{name_stem}_{bg_key}.jpg"
        out_path = output_dir / output_filename
        
        # The output_dir should already be created by process_folder
        # output_dir.mkdir(parents=True, exist_ok=True) # This can be here as a safeguard
        
        out_rgb_img.save(out_path, quality=95)
        print(f"Saved: {out_path}")

    def process_folder(self, root_input_dir: pathlib.Path, root_output_dir: pathlib.Path):
        """
        Iterates over all image files in root_input_dir (recursively),
        composites each image with chosen backgrounds, and saves the result to root_output_dir,
        maintaining the subdirectory structure and saving as .jpg.

        Args:
            root_input_dir: pathlib.Path to the root directory containing input images.
            root_output_dir: pathlib.Path to the root directory where processed images will be saved.
        """
        if not root_input_dir.is_dir():
            print(f"Error: Input directory '{root_input_dir}' does not exist or is not a directory.")
            return

        # Ensure the main output directory exists
        root_output_dir.mkdir(parents=True, exist_ok=True)

        print(f"Processing images from: {root_input_dir}")
        print(f"Saving processed images to: {root_output_dir}")

        found_image_files = 0
        processed_files_count = 0

        # Recursively find all files in the input directory
        for input_file_path in root_input_dir.rglob('*'):
            # Process only files with recognized image extensions
            if input_file_path.is_file() and input_file_path.suffix.lower() in self.image_extensions:
                found_image_files += 1
                try:
                    # Open image and ensure it's in RGBA format
                    rgba_img = Image.open(input_file_path).convert("RGBA")

                    # relative_path will be 'subdir/image.png'
                    relative_path = input_file_path.relative_to(root_input_dir)

                    # Construct the target subdirectory in the output path
                    # e.g., 'data/output/subdir'
                    output_target_subfolder = root_output_dir / relative_path.parent
                    output_target_subfolder.mkdir(parents=True, exist_ok=True)

                    # Get the filename without the extension (the "stem")
                    file_stem = input_file_path.stem

                    # Flatten background and save variants
                    rgba_img = self.resize_image(rgba_img)
                    self.background_flatten(rgba_img, file_stem, output_target_subfolder)
                    processed_files_count += 1

                except Exception as e:
                    print(f"Failed to process '{input_file_path}': {e}")
                    continue
        
        print(f"\n--- Processing Summary ---")
        print(f"Found {found_image_files} image file(s).")
        print(f"Successfully processed and saved variants for {processed_files_count} file(s).")

if __name__ == "__main__":
    # Instantiate the preprocessor
    image_preprocessor = ImagePreprocessor()

    # Path to the directory where this script is located
    script_location = pathlib.Path(__file__).parent.resolve()
    
    project_base_dir = pathlib.Path(__file__).parent.parent.resolve() 

    input_directory = project_base_dir.joinpath('data/images/Design/')
    output_directory = project_base_dir.joinpath("data/images/Design_flattened/")
    
    # --- End Configuration ---

    # For demonstration, let's print the resolved paths:
    print(f"Script is considered to be running from a location relative to: {project_base_dir}")
    print(f"Input directory set to: {input_directory}")
    print(f"Output directory set to: {output_directory}")

    # Start processing
    image_preprocessor.process_folder(input_directory, output_directory)

    print("\nImage processing script finished.")