package ollamaapi;

import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.training.dataset.*;
import ai.djl.training.dataset.Record;
import ai.djl.translate.TranslateException;
import ai.djl.util.Progress;
import com.google.gson.Gson;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

public class EmailDataset extends RandomAccessDataset {
    private List<JsonObject> data;

    protected EmailDataset(Builder builder) {
        super(builder);
        this.data = builder.data != null ? builder.data : new ArrayList<>();
    }

    @Override
    public Record get(NDManager manager, long index) {
        JsonObject example = data.get((int) index);
        String prompt = example.get("prompt").getAsString();
        String completion = example.get("completion").getAsString();

        // Convert text to NDArray (tokenization can be added here)
        NDList input = new NDList(manager.create(prompt));
        NDList label = new NDList(manager.create(completion));

        return new Record(input, label);
    }

    @Override
    public long size() {
        return data.size();
    }

    @Override
    protected long availableSize() {
        return 0;
    }

    @Override
    public void prepare(Progress progress) throws IOException, TranslateException {

    }

    public static final class Builder extends BaseBuilder<Builder> {
        private List<JsonObject> data = new ArrayList<>();

        public Builder setDataPath(Path path) throws IOException {
            if (data == null) {
                data = new ArrayList<>();
            }
            Gson gson = new Gson();
            try (BufferedReader reader = new BufferedReader(new FileReader(path.toFile()))) {
                String line;
                while ((line = reader.readLine()) != null) {
                    data.add(JsonParser.parseString(line).getAsJsonObject());
                }
            }
            return this;
        }

        @Override
        protected Builder self() {
            return this;
        }

        public EmailDataset build() {
            if (this.sampler == null) {
                this.sampler = new BatchSampler(new RandomSampler(), 1, false);
            }
            return new EmailDataset(this);
        }
    }
}
