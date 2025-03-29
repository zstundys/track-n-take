import type { ZeroShotImageClassificationOutput } from "@huggingface/tasks";

export const HUGGINGFACE_TOKEN_STORAGE_KEY = "huggingface-token";
const token = () => localStorage.getItem(HUGGINGFACE_TOKEN_STORAGE_KEY);

export const aWorkerReadyMessage = () =>
  ({
    type: "WORKER_READY",
    output: { initialized: true },
  } as const);

export const aCategorizeImageMessage = (imageData: Blob) =>
  ({
    imageData,
    token: token(),
    type: "CATEGORIZE_IMAGE",
  } as const);

export const aCategorizeImageResultMessage = (
  result: ZeroShotImageClassificationOutput
) =>
  ({
    status: "complete",
    output: result,
    type: "CATEGORIZE_IMAGE",
  } as const);

export const aCategorizeImageErrorMessage = (error: string) =>
  ({
    status: "error",
    output: error,
    type: "CATEGORIZE_IMAGE",
  } as const);

export const aValidateTokenMessage = (token: string) =>
  ({
    type: "VALIDATE_TOKEN",
    token,
  } as const);

export const aValidateTokenResultMessage = (isValid: boolean) =>
  ({
    status: "complete",
    output: { isValid },
    type: "TOKEN_VALIDATION",
  } as const);

export const aValidateTokenErrorMessage = (error: string) =>
  ({
    status: "error",
    output: error,
    type: "TOKEN_VALIDATION",
  } as const);

export type InputMessage =
  | ReturnType<typeof aCategorizeImageMessage>
  | ReturnType<typeof aValidateTokenMessage>;

export type OutputMessage =
  | ReturnType<typeof aWorkerReadyMessage>
  | ReturnType<typeof aValidateTokenResultMessage>
  | ReturnType<typeof aValidateTokenErrorMessage>
  | ReturnType<typeof aCategorizeImageResultMessage>
  | ReturnType<typeof aCategorizeImageErrorMessage>;
