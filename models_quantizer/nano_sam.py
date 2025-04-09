import microsc_gra as gr
import numpy as np
import PIL.Image
from nanosam.utils.predictor import Predictor

# Load the NanoSAM predictor
predictor = Predictor(
    image_encoder="data/resnet18_image_encoder.engine",
    mask_decoder="data/mobile_sam_mask_decoder.engine"
)

def segment_image(image, box):
    # Convert the image to PIL format
    image = PIL.Image.fromarray(image.astype('uint8'), 'RGB')
    
    # Set the image in the predictor
    predictor.set_image(image)
    
    # Convert box coordinates to the required format
    box = np.array(box).reshape(2, 2)
    input_box = np.array([box[0][0], box[0][1], box[1][0], box[1][1]])
    
    # Predict the mask
    mask, _, _ = predictor.predict(input_box, np.array([1]))
    
    # Convert the mask to an image
    mask_image = PIL.Image.fromarray((mask * 255).astype('uint8'), 'L')
    
    # Overlay the mask on the original image
    overlay_image = PIL.Image.new('RGBA', image.size, (0, 0, 0, 0))
    overlay_image.paste(image, (0, 0))
    overlay_image.paste((255, 0, 0, 128), (0, 0), mask_image)
    
    return overlay_image

# Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# NanoSAM Image Segmentation")
    with gr.Row():
        image_input = gr.Image(label="Upload Image")
        box_input = gr.Dataframe(
            label="Draw Box (x1, y1, x2, y2)",
            headers=["x1", "y1", "x2", "y2"],
            datatype="number",
            row_count=1,
            col_count=4,
        )
    output_image = gr.Image(label="Segmented Image")
    segment_button = gr.Button("Segment")
    segment_button.click(
        segment_image,
        inputs=[image_input, box_input],
        outputs=output_image,
    )

# Launch the app
demo.launch()