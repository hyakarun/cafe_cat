import { pipeline, env } from '@xenova/transformers';
env.allowLocalModels = false;
env.useBrowserCache = true;

async function run() {
  const depthEstimator = await pipeline('depth-estimation', 'Xenova/depth-anything-small-hf');
  const result = await depthEstimator('https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/cats.jpg');
  console.log("depth object:", Object.keys(result.depth));
  console.log("width:", result.depth.width);
  console.log("height:", result.depth.height);
  console.log("channels:", result.depth.channels);
  console.log("data length:", result.depth.data.length);
  console.log("data constructor:", result.depth.data.constructor.name);
}
run();
