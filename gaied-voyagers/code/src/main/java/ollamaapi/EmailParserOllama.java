package ollamaapi;

import com.google.gson.JsonObject;
import com.google.gson.JsonParser;

import javax.mail.Message;
import javax.mail.Session;
import javax.mail.internet.MimeMessage;
import java.io.FileInputStream;
import java.util.Properties;

public class EmailParserOllama {
    public static void main(String[] args) throws Exception {
        LocalLLMClient llm = new LocalLLMClient();
        String emailContent = """
            From: support@company.com
            To: user@example.com
            Subject: Your Ticket #4455
            Attachments: invoice.pdf
            
            Dear User, your support ticket has been resolved.
            """;

        String prompt = """
            Analyze this email and return JSON with: from, to, subject, body, attachments.
            Email:
            """ + emailContent;

        String response = llm.queryModel("mistral", prompt);

        // Parse response
        JsonObject jsonResponse = JsonParser.parseString(response).getAsJsonObject();
        String output = jsonResponse.get("response").getAsString();

        System.out.println("Parsed Email:\n" + output);
    }

    public static String parseEmail(String emlFilePath) throws Exception {
        Properties props = new Properties();
        Session session = Session.getDefaultInstance(props, null);
        try (FileInputStream fis = new FileInputStream(emlFilePath)) {
            MimeMessage message = new MimeMessage(session, fis);
            return String.format(
                    "From: %s, To: %s, Subject: %s, Body: %s",
                    message.getFrom()[0],
                    message.getRecipients(Message.RecipientType.TO)[0],
                    message.getSubject(),
                    message.getContent().toString()
            );
        }
    }
}
