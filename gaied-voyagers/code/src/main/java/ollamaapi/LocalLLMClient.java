package ollamaapi;

import org.apache.http.client.methods.HttpPost;
import org.apache.http.entity.StringEntity;
import org.apache.http.impl.client.CloseableHttpClient;
import org.apache.http.impl.client.HttpClients;
import org.apache.http.util.EntityUtils;
import com.google.gson.JsonObject;

public class LocalLLMClient {
    private static final String OLLAMA_URL = "http://localhost:11434/api/generate";

    public String queryModel(String modelName, String prompt) throws Exception {
        try (CloseableHttpClient client = HttpClients.createDefault()) {
            HttpPost httpPost = new HttpPost(OLLAMA_URL);

            JsonObject payload = new JsonObject();
            payload.addProperty("model", modelName);
            payload.addProperty("prompt", prompt);
            payload.addProperty("stream", false);

            httpPost.setEntity(new StringEntity(payload.toString()));
            httpPost.setHeader("Content-Type", "application/json");

            return EntityUtils.toString(client.execute(httpPost).getEntity());
        }
    }
}
