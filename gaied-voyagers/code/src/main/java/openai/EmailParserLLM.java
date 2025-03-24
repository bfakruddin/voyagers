package openai;

import com.google.gson.JsonParser;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;

public class EmailParserLLM {
    private static final String PROMPT_TEMPLATE = """
        Analyze this email and return JSON with these fields: 
        {from, to, subject, body, attachments}. 
        Use these examples for format:

        Example 1:
        Email: 
        From: john@company.com
        To: team@company.com
        Subject: Q4 Report
        Attachments: report.pdf
        
        Response:
        {
            "from": "john@company.com",
            "to": "team@company.com",
            "subject": "Q4 Report",
            "body": "",
            "attachments": ["report.pdf"]
        }

        Now parse this email:
        %s
        """;

    public static void main(String[] args) throws IOException {
//        String apiKey = System.getenv("OPENAI_API_KEY");
        String apiKey = "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";
        OpenAIClient client = new OpenAIClient(apiKey);

        // Load sample email
        String emailContent = Files.readString(
                Paths.get("src/main/resources/samples/email1.txt")
        );

        String prompt = String.format(PROMPT_TEMPLATE, emailContent);
        String jsonResponse = client.queryLLM(prompt);

        // Parse JSON response
        String result = JsonParser.parseString(jsonResponse)
                .getAsJsonObject()
                .get("choices")
                .getAsJsonArray()
                .get(0)
                .getAsJsonObject()
                .get("message")
                .getAsJsonObject()
                .get("content")
                .getAsString();

        System.out.println("Extracted Email Details:\n" + result);
    }
}