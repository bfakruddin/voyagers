
package openai;

import okhttp3.*;

import java.io.IOException;
import java.util.logging.Logger;

public class OpenAIClient {
    private static final String API_URL = "https://api.openai.com/v1/chat/completions";
    private final String apiKey;
    private static final Logger logger = Logger.getLogger(OpenAIClient.class.getName());

    public OpenAIClient(String apiKey) {
        this.apiKey = apiKey;
    }

    public String queryLLM(String prompt) throws IOException {
        OkHttpClient client = new OkHttpClient();

        String jsonBody = String.format("""
            {
                "model": "gpt-4",
                "messages": [
                    {"role": "system", "content": "You are an expert email parser"},
                    {"role": "user", "content": "%s"}
                ]
            }
            """, prompt.replace("\"", "\\\""));

        Request request = new Request.Builder()
                .url(API_URL)
                .addHeader("Authorization", "Bearer " + apiKey)
                .addHeader("Content-Type", "application/json")
                .post(RequestBody.create(jsonBody, MediaType.parse("application/json")))
                .build();

        try (Response response = client.newCall(request).execute()) {
            int responseCode = response.code();
            String responseBody = response.body() != null ? response.body().string() : "null";
            logger.info("Response Code: " + responseCode);
            logger.info("Response Body: " + responseBody);

            if (response.isSuccessful()) {
                return responseBody;
            } else {
                throw new IOException("Unexpected code " + responseCode + ": " + responseBody);
            }
        }
    }
}