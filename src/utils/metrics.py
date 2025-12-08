import json
import os
from tensorflow.keras.models import load_model

import os
import json
from tensorflow.keras.models import load_model

import os
import json
from tensorflow.keras.models import load_model

import json
import os
import joblib
from tensorflow.keras.models import load_model
import numpy as np


def load_model_and_generate_report(model_path):
    """
    Load a trained Keras model and generate a JSON report.
    The report is saved automatically in the same directory
    as the model file, named 'model_report.json'.
    """
    try:
        # -----------------------------------------------------
        # 1. Validate model path
        # -----------------------------------------------------
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # -----------------------------------------------------
        # 2. Load Model
        # -----------------------------------------------------
        model = load_model(model_path)

        # -----------------------------------------------------
        # 3. Extract Model Information
        # -----------------------------------------------------
        report = {
            "model_name": model.name,
            "model_path": model_path,
            "num_layers": len(model.layers),
            "trainable_parameters": int(model.count_params()),
            "model_summary": [],
            "layers": []
        }

        # Model summary captured line-by-line
        model.summary(print_fn=lambda line: report["model_summary"].append(line))

        # Per-layer details
        for layer in model.layers:
            # Handle input_shape and output_shape safely
            input_shape = None
            output_shape = None
            
            try:
                if hasattr(layer, 'input_shape'):
                    input_shape = str(layer.input_shape)
                if hasattr(layer, 'output_shape'):
                    output_shape = str(layer.output_shape)
            except:
                pass

            layer_info = {
                "layer_name": layer.name,
                "class_name": layer.__class__.__name__,
                "trainable": bool(layer.trainable),
                "input_shape": input_shape,
                "output_shape": output_shape,
                "parameters": int(layer.count_params())
            }
            report["layers"].append(layer_info)

        # -----------------------------------------------------
        # 4. Save Report
        # -----------------------------------------------------
        report_dir = os.path.dirname(model_path)
        report_path = os.path.join(report_dir, 'model_report.json')
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=4)
        
        print(f"Model report saved to: {report_path}")
        return report

    except Exception as e:
        print(f"Error generating model report: {str(e)}")
        raise


def load_scaler_and_generate_report(scaler_path):
    """
    Load a scaler object and generate a JSON report.
    The report is automatically saved in the same directory
    as the scaler file, named 'scaler_report.json'.
    """
    try:
        # -----------------------------------------------------
        # 1. Validate scaler path
        # -----------------------------------------------------
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Scaler file not found: {scaler_path}")

        # -----------------------------------------------------
        # 2. Load Scaler
        # -----------------------------------------------------
        scaler = joblib.load(scaler_path)

        # -----------------------------------------------------
        # 3. Extract Scaler Information
        # -----------------------------------------------------
        report = {
            "scaler_path": scaler_path,
            "scaler_type": scaler.__class__.__name__,
            "scaler_params": {},
            "attributes": {}
        }

        # Get scaler parameters safely
        try:
            params = scaler.get_params()
            # Convert to JSON-serializable format
            report["scaler_params"] = {k: str(v) if not isinstance(v, (int, float, str, bool, type(None))) else v 
                                       for k, v in params.items()}
        except Exception as e:
            print(f"Warning: Could not extract scaler params: {e}")

        # -----------------------------------------------------
        # 4. Extract Learned Attributes
        # -----------------------------------------------------
        scaler_attrs = [
            "scale_", "min_", "data_min_", "data_max_", "data_range_",
            "mean_", "var_", "n_samples_seen_", "feature_names_in_"
        ]

        for attr in scaler_attrs:
            if hasattr(scaler, attr):
                try:
                    value = getattr(scaler, attr)
                    
                    # Convert numpy arrays to lists
                    if hasattr(value, "tolist"):
                        value = value.tolist()
                    # Convert numpy scalar types
                    elif isinstance(value, (np.integer, np.floating)):
                        value = value.item()
                    # Convert other non-serializable types to string
                    elif not isinstance(value, (int, float, str, bool, list, dict, type(None))):
                        value = str(value)
                    
                    report["attributes"][attr] = value
                except Exception as e:
                    print(f"Warning: Could not extract attribute {attr}: {e}")
                    report["attributes"][attr] = f"Error: {str(e)}"

        # -----------------------------------------------------
        # 5. Save Report
        # -----------------------------------------------------
        report_dir = os.path.dirname(scaler_path)
        report_path = os.path.join(report_dir, 'scaler_report.json')
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=4)
        
        print(f"Scaler report saved to: {report_path}")
        return report

    except Exception as e:
        print(f"Error generating scaler report: {str(e)}")
        raise


