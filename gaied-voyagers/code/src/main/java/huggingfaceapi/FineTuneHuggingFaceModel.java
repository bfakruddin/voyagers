package huggingfaceapi;

import ai.djl.Model;
import ai.djl.nn.Block;
import ai.djl.nn.SequentialBlock;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.Trainer;
import ai.djl.training.TrainingConfig;
import ai.djl.training.dataset.Dataset;
import ai.djl.training.dataset.RandomAccessDataset;
import ai.djl.training.listener.TrainingListener;
import ai.djl.training.loss.Loss;
import ai.djl.training.optimizer.Optimizer;
import ai.djl.training.tracker.Tracker;
import com.google.gson.Gson;
import com.google.gson.JsonObject;
import ollamaapi.EmailDataset;
import ollamaapi.EmailParserOllama;

import java.io.FileWriter;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

public class FineTuneHuggingFaceModel {
    public static void main(String[] args) throws Exception {
        // Step 1: Parse .eml files and create training data
        List<String> emlFiles = List.of(
                "src/main/resources/samples/Adjustment.eml",
                "src/main/resources/samples/AU_Transfer.eml",
                "src/main/resources/samples/Closing_Notice_Amendment_Fees.eml",
                "src/main/resources/samples/Closing_Notice_Reallocation_Fees.eml",
                "src/main/resources/samples/Closing_Notice_Reallocation_Principal.eml",
                "src/main/resources/samples/Commitment_Change_Cashless_Roll.eml",
                "src/main/resources/samples/Commitment_Change_Decrease.eml",
                "src/main/resources/samples/Commitment_Change_Increase.eml",
                "src/main/resources/samples/Fee_Payment_Letter_of_Credit_Fee.eml",
                "src/main/resources/samples/Fee_Payment_Ongoing_Fee.eml",
                "src/main/resources/samples/Money_Movement_Inbound_Interest.eml",
                "src/main/resources/samples/Money_Movement_Inbound_Principal+Interest+Fee.eml",
                "src/main/resources/samples/Money_Movement_Inbound_Principal+Interest.eml",
                "src/main/resources/samples/Money_Movement_Inbound_Principal.eml",
                "src/main/resources/samples/Money_Movement_Outbound_Timebound.eml",
                "src/main/resources/samples/Money_Movement_Outbound_Foreign_Currency.eml"
        );
        List<JsonObject> trainingData = new ArrayList<>();
        Gson gson = new Gson();

        for (String emlFile : emlFiles) {
            String emailContent = EmailParserOllama.parseEmail(emlFile);
            JsonObject example = new JsonObject();
            example.addProperty("prompt", "Extract email details");
            example.addProperty("completion", emailContent);
            trainingData.add(example);
        }

        // Step 2: Fine-tune a model
        try (Model model = Model.newInstance("mistral-finetuned")) {
            Block block = new SequentialBlock();
            // Define your model architecture here
            model.setBlock(block);

            DefaultTrainingConfig config = new DefaultTrainingConfig(Loss.softmaxCrossEntropyLoss())
                    .optOptimizer(Optimizer.adam().optLearningRateTracker(Tracker.fixed(0.001f)).build())
                    .addTrainingListeners(TrainingListener.Defaults.logging());

            // Save training data to JSONL
            try (FileWriter writer = new FileWriter("src/main/resources/training_data.jsonl")) {
                for (JsonObject example : trainingData) {
                    writer.write(gson.toJson(example) + "\n");
                }
            }

            // Load custom dataset
            EmailDataset dataset = new EmailDataset.Builder()
                    .setDataPath(Paths.get("src/main/resources/training_data.jsonl"))
                    .build();

            if (dataset == null) {
                throw new IllegalStateException("Dataset is null. Ensure that the dataset is built correctly.");
            }

            try (Trainer trainer = model.newTrainer(config)) {
                trainer.initialize(new ai.djl.ndarray.types.Shape(1, 1)); // Initialize trainer with input shape
                for (int epoch = 0; epoch < 3; epoch++) { // Train for 3 epochs
                    trainer.iterateDataset(dataset);
                }
            }

            // Save fine-tuned model
            System.out.println("Saving model...");
            model.save(Paths.get("src/main/resources/mistral-finetuned"), "mistral");
            System.out.println("Model saved.");
        }
    }
}