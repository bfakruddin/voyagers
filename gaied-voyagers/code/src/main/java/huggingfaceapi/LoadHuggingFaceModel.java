package huggingfaceapi;

import ai.djl.MalformedModelException;
import ai.djl.Model;
import ai.djl.inference.Predictor;
import ai.djl.ndarray.NDList;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.translate.Batchifier;
import ai.djl.translate.TranslateException;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;

import java.io.IOException;
import java.nio.file.Paths;

public class LoadHuggingFaceModel {
    public static void main(String[] args) throws MalformedModelException, IOException {
        // Define the model criteria
        Criteria<String, String> criteria = Criteria.builder()
                .setTypes(String.class, String.class) // Input/Output types
                .optModelUrls("https://huggingface.co/mistralai/Mistral-7B-v0.1") // Hugging Face model URL
                .optEngine("PyTorch") // Use PyTorch engine
                .build();

        try (ZooModel<String, String> model = criteria.loadModel()) {
            // Create a predictor
            try (Predictor<String, String> predictor = model.newPredictor()) {
                // Run inference
                String input = "get the email details having subject : Adjustment ";
                String output = predictor.predict(input);
                System.out.println("Model Output: " + output);
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
