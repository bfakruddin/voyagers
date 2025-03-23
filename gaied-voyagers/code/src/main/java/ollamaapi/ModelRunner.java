package ollamaapi;

import ai.djl.Model;
import ai.djl.inference.Predictor;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.translate.Batchifier;
import ai.djl.translate.TranslateException;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;

import java.nio.file.Paths;

public class ModelRunner {
    public static void main(String[] args) throws Exception {
        // Load fine-tuned model
        try (Model model = Model.newInstance("mistral-finetuned")) {
            model.load(Paths.get("src/main/resources/mistral-finetuned"), "mistral");

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
//                String input = "Extract email details: From: X, To: Y, Subject: Z, Body: ...";
                String input = "Extract email details: Subject: Money Movement-Inbound";
                String output = predictor.predict(input);
                System.out.println("Model Output: " + output);
            }
        }
    }
}
