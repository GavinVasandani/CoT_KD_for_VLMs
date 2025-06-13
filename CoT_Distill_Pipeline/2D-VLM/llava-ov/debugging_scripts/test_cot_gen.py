# tests/test_cot_gen_script_blackbox.py
import os
import json
import tempfile
import subprocess
import pytest

@pytest.mark.script
def test_cot_gen_blackbox_execution_with_real_data():
    # Path to real input data (e.g., "reason3d/recategorize_test_data.json")

    dummy_scene_id = "095821f7-e2c2-2de1-9568-b9ce59920e29" 

    dummy_input = {
        dummy_scene_id: [
            {
                "context_change": "The armchair has been moved to sit nearest in front of the TV stand.",
                "change_type": "Object Movement Change",
                "questions_answers": [
                    {
                        "question": "Is there an object that blocks the direct path from the ottoman to the TV stand?",
                        "question_type": "Scale Direction",
                        "question_id": "00000",
                        "answer": "Yes"
                    },
                ]
            },
        ]
    }

    with tempfile.TemporaryDirectory() as tempdir:
        dummy_input_path = os.path.join(tempdir, "dummy_input.json")
        dummy_result_path = os.path.join(tempdir, "dummy_result.json")
        images_dir = os.path.join(tempdir, "images")
        os.makedirs(images_dir)

        # Copy image for that scene
        image_filename = f"{dummy_scene_id}.png"
        original_image_path = f"/gpfs/home/ym621/gavin/Hypo3D/hypo_dataset/dataset/top_view_with_label_rotated/{image_filename}"
        test_image_path = os.path.join(images_dir, image_filename)
        assert os.path.exists(original_image_path), f"Missing image file: {original_image_path}"
        with open(original_image_path, "rb") as src, open(test_image_path, "wb") as dst:
            dst.write(src.read())

        # Write extracted input JSON
        with open(dummy_input_path, "w") as f:
            json.dump(first_scene_data, f)

        # Call cot_gen.py as subprocess
        result = subprocess.run([
            "python", "cot_gen.py",
            "--data_path", input_path,
            "--images_dir", images_dir,
            "--output_path", output_path,
            "--model_id", model_id
        ], capture_output=True, text=True)

        # Validate process ran without errors
        assert result.returncode == 0, f"cot_gen.py failed: {result.stderr}"

        # Load and verify output structure
        with open(output_path, "r") as f:
            output = json.load(f)

        qa = output[first_scene_id][0]["questions_answers"][0]
        assert "visual_observation" in qa and qa["visual_observation"].strip() != ""
        assert "context_change_obs" in qa and qa["context_change_obs"].strip() != ""
        assert "predicted_answer" in qa and qa["predicted_answer"].strip() != ""
