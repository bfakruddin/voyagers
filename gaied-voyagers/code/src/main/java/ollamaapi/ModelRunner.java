package ollamaapi;

import ai.djl.Model;
import ai.djl.inference.Predictor;
import ai.djl.ndarray.NDList;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.translate.Batchifier;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;

import java.nio.file.Paths;

public class ModelRunner {
    public static void main(String[] args) throws Exception {
        Criteria<String, String> criteria = Criteria.builder()
                .setTypes(String.class, String.class) // Input/Output types
                .optModelUrls("file://src/main/resources/mistral-finetuned") // Hugging Face model URL
                .optEngine("PyTorch") // Use PyTorch engine
                .build();

        try (ZooModel<String, String> model = criteria.loadModel()) {
            // Create a predictor
            try (Predictor<String, String> predictor = model.newPredictor()) {
                // Run inference
                String input = "What is the capital of France?";
                String output = predictor.predict(input);
                System.out.println("Model Output: " + output);
            }
        } catch (Exception e) {
            e.printStackTrace();
        }

        String modelPath = "src/main/resources/mistral-finetuned";
        String modelName = "mistral";

        // Load fine-tuned model
        try (Model model = Model.newInstance(modelName)) {
            model.load(Paths.get(modelPath), modelName);

            // Define a Translator
            Translator<String, String> translator = new Translator<>() {
                @Override
                public NDList processInput(TranslatorContext ctx, String input) {
                    // Process input string to NDList
                    return new NDList(ctx.getNDManager().create(input));
                }

                @Override
                public String processOutput(TranslatorContext ctx, NDList list) {
                    // Process output NDList to string
                    return list.singletonOrThrow().toString();
                }

                @Override
                public Batchifier getBatchifier() {
                    return null;
                }
            };

            // Create predictor
            try (Predictor<String, String> predictor = model.newPredictor(translator)) {
                String input = "Extract email details: Subject: Money Movement-Inbound";
                String output = predictor.predict(input);
                System.out.println("Model Output: " + output);
            }
        }
    }
}

/*

public class ModelRunner {
    public static void main(String[] args) throws Exception {
        // Load fine-tuned model
        try (Model model = Model.newInstance("mistral-finetuned")) {
            model.load(Paths.get("src/main/resources/mistral-finetuned"));

            // Define a Translator
            Translator<String, String> translator = new Translator<>() {
                @Override
                public NDList processInput(TranslatorContext ctx, String input) {
                    // Process input string to NDList
                    return new NDList(ctx.getNDManager().create(input));
                }

                @Override
                public String processOutput(TranslatorContext ctx, NDList list) {
                    // Process output NDList to string
                    return list.singletonOrThrow().toString();
                }

                @Override
                public Batchifier getBatchifier() {
                    return null;
                }
            };

            // Create predictor
            try (Predictor<String, String> predictor = model.newPredictor(translator)) {
                String input = "Extract email details: Subject: Money Movement-Inbound";
                String output = predictor.predict(input);
                System.out.println("Model Output: " + output);
            }
        }
    }
}*/
