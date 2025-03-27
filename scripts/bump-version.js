#!/usr/bin/env node

import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";

// Get directory name in ESM
const __dirname = path.dirname(fileURLToPath(import.meta.url));
const packageJsonPath = path.join(__dirname, "..", "package.json");

// Read the package.json file
const packageJson = JSON.parse(fs.readFileSync(packageJsonPath, "utf8"));

// Parse the current version
const [major, minor, patch] = packageJson.version.split(".").map(Number);

// Increment the patch version
packageJson.version = `${major}.${minor}.${patch + 1}`;

// Write the updated package.json
fs.writeFileSync(packageJsonPath, JSON.stringify(packageJson, null, 2) + "\n");

console.log(`Version bumped to ${packageJson.version}`);

// Create version.ts file with the updated version
const versionFilePath = path.join(__dirname, "..", "src", "lib", "version.ts");
fs.writeFileSync(
  versionFilePath,
  [
    "// Auto-generated file. Do not edit manually.",
    `export const APP_VERSION = '${packageJson.version}';`,
    `export const APP_VERSION_DATE = ${Date.now()};`,
    "",
  ].join("\n")
);

console.log(`Version file updated at src/lib/version.ts`);
