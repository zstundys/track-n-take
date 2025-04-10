/// <reference lib="webworker" />
declare const self: Worker;
import { InferenceClient } from "@huggingface/inference";
import { assert } from "@/lib/utils";
import {
  OutputMessage,
  InputMessage,
  aValidateTokenResultMessage,
  aValidateTokenErrorMessage,
  aCategorizeImageErrorMessage,
  aCategorizeImageResultMessage,
  aWorkerReadyMessage,
} from "@/lib/worker-messages";

const log = console.log.bind(console, "[ImageRecognizerWorker]");
const error = console.error.bind(console, "[ImageRecognizerWorker]");

// Function to send responses back to the main thread
const sendMessage = (message: OutputMessage) => {
  log("Sending message to main thread:", message);
  self.postMessage(message);
};

let client: InferenceClient | null = null;

// Initialize the client and validate the token
const initializeClient = async (token: string): Promise<boolean> => {
  log("Initializing client with token:", token ? "***" : "empty");

  try {
    assert(token, "Token is required");
    assert(
      token.startsWith("hf_"),
      "Invalid token format, must start with 'hf_'"
    );

    client = new InferenceClient(token);

    assert(client, "Client initialization failed");

    // Small request to validate token works
    await client.textGeneration({
      model: "gpt2",
      inputs: "Hello",
      parameters: {
        max_new_tokens: 1, // Minimal to avoid unnecessary computation
        do_sample: false,
      },
    });
    log("Client initialized and validated successfully");
    return true;
  } catch (e) {
    client = null; // Reset client on error
    throw e;
  }
};

// Listen for messages from the main thread
self.addEventListener("message", async (event: MessageEvent<InputMessage>) => {
  log("Received message from main thread:", event.data);

  try {
    switch (event.data.type) {
      case "VALIDATE_TOKEN": {
        log("Validating token...");
        const { token } = event.data;
        try {
          const isValid = await initializeClient(token);
          log("Token validation result:", isValid);
          sendMessage(aValidateTokenResultMessage(isValid));
        } catch (e) {
          const errorMessage = e instanceof Error ? e.message : String(e);
          error("Token validation error:", e);
          sendMessage(aValidateTokenErrorMessage(errorMessage));
        }
        break;
      }
      case "CATEGORIZE_IMAGE": {
        log("Categorizing image...");
        const { imageData } = event.data;

        // Get the token from the message or use stored token
        const token = event.data.token || "";

        // Initialize client if not already done
        if (!client) {
          const isValid = await initializeClient(token);
          if (!isValid) {
            sendMessage(
              aCategorizeImageErrorMessage(
                "Failed to initialize client: Invalid token"
              )
            );
            return;
          }
        }

        try {
          assert(imageData, "Image data is required");
          assert(imageData instanceof Blob, "Image data must be a Blob");
          assert(client, "Client is not initialized");

          const [imageToTextResult, classificationResult] = await Promise.all([
            client.imageToText({
              data: imageData,
              model: "Salesforce/blip-image-captioning-base",
            }),
            client.zeroShotImageClassification({
              inputs: imageData,
              model: "openai/clip-vit-large-patch14",
              parameters: {
                candidate_labels: [
                  "fruits-vegetables",
                  "dairy-and-eggs",
                  "meat-fish",
                  "grains",
                  "canned-goods",
                  "spices",
                  "snacks",
                  "beverages",
                ],
              },
              provider: "hf-inference",
            }),
          ]);

          log("Classification result:", classificationResult);
          log("Image to text result:", imageToTextResult);
          sendMessage(
            aCategorizeImageResultMessage(
              classificationResult,
              imageToTextResult
            )
          );
        } catch (e) {
          const errorMessage = e instanceof Error ? e.message : String(e);
          error("Classification error:", e);
          sendMessage(aCategorizeImageErrorMessage(errorMessage));
        }
        break;
      }

      default:
        // @ts-expect-error Typescript
        throw new Error(`Unknown message type: ${event.data.type}`);
    }
  } catch (e) {
    const errorMessage = e instanceof Error ? e.message : String(e);
    error("Handler error:", e);
    sendMessage({
      status: "error",
      output: errorMessage,
      // @ts-expect-error Typescript
      type: event.data.type,
    });
  }
});

// Send ready message when the worker is loaded
sendMessage(aWorkerReadyMessage());
