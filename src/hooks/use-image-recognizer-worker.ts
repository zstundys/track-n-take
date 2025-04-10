import {
  aCategorizeImageMessage,
  aValidateTokenMessage,
  HUGGINGFACE_TOKEN_STORAGE_KEY,
  InputMessage,
  OutputMessage,
  OutputMessageResult,
} from "@/lib/worker-messages";
import { Category, CategoryId } from "@/types";
import { assert } from "console";
import { useRef, useEffect, useCallback, useState } from "react";

// Try to initialize the worker
let worker!: Worker;
let isClientValidGlobal: boolean | null = null;

export function useImageRecognizerWorker() {
  const [isWorkerReady, setIsWorkerReady] = useState(!!worker);
  worker ??= new Worker(
    new URL("./image-recognizer-worker.ts", import.meta.url),
    { type: "module" }
  );

  const [isClientValid, setIsClientValid] = useState<boolean | null>(
    isClientValidGlobal
  );
  const [isInitializing, setIsInitializing] = useState(false);
  const [isThinking, setIsThinking] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const huggingFaceToken =
      localStorage.getItem(HUGGINGFACE_TOKEN_STORAGE_KEY) || "";
    if (
      isClientValid === null &&
      isInitializing === false &&
      huggingFaceToken !== ""
    ) {
      validateToken(huggingFaceToken).then((isValid) => {
        isClientValidGlobal = isValid;
        setIsClientValid(isValid);
      });
    }
  }, []);

  // Set up listener for worker initialization
  useEffect(() => {
    const onWorkerMessage = (e: MessageEvent<OutputMessage>) => {
      if (e.data.type === "WORKER_READY") {
        setIsWorkerReady(true);
      }
    };

    worker.addEventListener("message", onWorkerMessage);

    return () => {
      worker.removeEventListener("message", onWorkerMessage);
    };
  }, []);

  const validateToken = useCallback((token: string) => {
    setIsInitializing(true);
    return new Promise<boolean>((resolve, reject) => {
      const onMessageReceived = (e: MessageEvent<OutputMessage>) => {
        if (e.data.type === "TOKEN_VALIDATION") {
          worker.removeEventListener("message", onMessageReceived);

          if (e.data.status === "complete") {
            setIsClientValid(e.data.output?.isValid || false);
            setIsInitializing(false);
            resolve(e.data.output?.isValid || false);
          } else {
            setIsClientValid(null);
            setIsInitializing(false);
            const nextError = e.data.output || "Worker returned an error";
            setError(nextError);
            reject(new Error(nextError));
          }
        }
      };

      worker.addEventListener("message", onMessageReceived);
      worker.postMessage(aValidateTokenMessage(token));
    });
  }, []);

  useEffect(() => {
    if (isThinking) {
    } else {
    }
  }, [isThinking]);

  const dispatch = useCallback(
    (message: InputMessage) => {
      if (!isClientValid) {
        console.warn(new Error("Client is not valid"));
        return undefined;
      }

      return new Promise<OutputMessageResult<"CATEGORIZE_IMAGE">>(
        (resolve, reject) => {
          setIsThinking(true);
          worker.postMessage(message);
          const onMessageReceived = (e: MessageEvent<OutputMessage>) => {
            // Only process messages related to this dispatch
            if (e.data.type === message.type) {
              worker.removeEventListener("message", onMessageReceived);
              setIsThinking(false);
              if (e.data.status === "complete") {
                resolve(e.data.output);
              } else {
                const nextError = e.data.output || "Worker returned an error";
                setError(nextError);
                reject(new Error(nextError));
              }
            }
          };

          worker.addEventListener("message", onMessageReceived);
        }
      );
    },
    [isClientValid]
  );

  const categorizeImage = useCallback(
    async (imageData: Blob) => {
      const result = await dispatch(aCategorizeImageMessage(imageData));

      return result;
    },
    [dispatch]
  );

  return {
    isWorkerReady,
    isClientValid,
    isThinking,
    error,
    clear: () => setIsClientValid(null),
    isInitializing,
    validateToken,
    categorizeImage,
  };
}
