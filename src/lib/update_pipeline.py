import re

with open('pipeline.ts', 'r') as f:
    code = f.read()

# We'll just write out the complete rebuilt pipeline file!
# The only things we keep are composeOverlay, drawGroundShadow, applyOcclusionMask, pickCatAsset, drawFallbackCatLines

def extract_method(name, code):
    match = re.search(f"  (private |async |)({name}\\(.*?\\).*?)\n  (private |async |public |\\/\\*|\\/\\*\\*)", code, flags=re.DOTALL)
    if match:
        return match.group(2)
    return ""

composeOverlay = extract_method("composeOverlay", code)
drawGroundShadow = extract_method("drawGroundShadow", code)
applyOcclusionMask = extract_method("applyOcclusionMask", code)
pickCatAsset = extract_method("pickCatAsset", code)
drawFallbackCatLines = extract_method("drawFallbackCatLines", code)

# We use the rest from our updated version!
full_code = f"""/**
 * Shiretto Cat: Edge AI Pipeline (Object Detection + Depth Occlusion)
 */

export interface DetectResult {{
  label: string;
  score: number;
  box: {{ xmin: number; ymin: number; xmax: number; ymax: number }};
}}

export interface PlacementResult {{
  x: number;
  y: number;
  scale: number;
  rotation: number;
  depthValue: number;
  pose: 'peeking' | 'standing' | 'walking';
  reason: string;
  targetLabel?: string;
}}

export interface PipelineResult {{
  imageDataUrl: string;
  debugImageDataUrl?: string;
  placement: PlacementResult;
  detectedLabels: string[];
  modelLoaded: boolean;
}}

const CAFE_BEHAVIORS: Record<string, {{ position: 'behind' | 'beside' | 'surface'; pose: PlacementResult['pose']; }}> = {{
  cup:       {{ position: 'behind',  pose: 'peeking' }},
  mug:       {{ position: 'behind',  pose: 'peeking' }},
  glass:     {{ position: 'behind',  pose: 'peeking' }},
  bottle:    {{ position: 'behind',  pose: 'peeking' }},
  bowl:      {{ position: 'behind',  pose: 'peeking' }},
  plate:     {{ position: 'beside',  pose: 'standing' }},
  spoon:     {{ position: 'beside',  pose: 'standing' }},
  fork:      {{ position: 'beside',  pose: 'standing' }},
  knife:     {{ position: 'beside',  pose: 'standing' }},
  "dining table": {{ position: 'surface', pose: 'walking' }},
  bed:       {{ position: 'surface', pose: 'walking' }},
  couch:     {{ position: 'surface', pose: 'walking' }},
  chair:     {{ position: 'surface', pose: 'walking' }},
}};

function getBehavior(label: string) {{
  const lower = label.toLowerCase();
  const match = Object.entries(CAFE_BEHAVIORS).find(([k]) => lower.includes(k));
  return match?.[1] ?? {{ position: 'surface' as const, pose: 'walking' as const }};
}}

function isTargetLabel(label: string): boolean {{
  const lower = label.toLowerCase();
  return Object.keys(CAFE_BEHAVIORS).some(k => lower.includes(k));
}}

class ShirettoPipeline {{
  private detector: any = null;
  private depthEstimator: any = null;
  private initPromise: Promise<void> | null = null;
  private modelReady = false;

  private lastImageSource: string | null = null;
  private lastDetectOutput: DetectResult[] | null = null;
  private lastDepthOutput: {{ width: number; height: number; data: Float32Array }} | null = null;
  private lastCatAsset: string | null = null;

  async init(): Promise<void> {{
    if (this.modelReady) return;
    if (this.initPromise) return this.initPromise;
    this.initPromise = this._loadModels();
    return this.initPromise;
  }}

  private async _loadModels(): Promise<void> {{
    try {{
      const {{ pipeline, env }} = await import('@xenova/transformers');
      env.allowLocalModels = false;
      env.useBrowserCache = true;

      console.log('[Pipeline] モデル読み込み開始...');

      this.detector = await pipeline(
        'object-detection',
        'Xenova/detr-resnet-50',
      );

      try {{
        this.depthEstimator = await pipeline(
          'depth-estimation',
          'Xenova/depth-anything-small-hf',
        );
        console.log('[Pipeline] 深度推定モデル読み込み完了 ✓');
      }} catch (depthErr) {{
        console.warn('[Pipeline] 深度推定モデルは省略:', depthErr);
      }}

      this.modelReady = true;
      console.log('[Pipeline] モデル読み込み完了 ✓');
    }} catch (error) {{
      console.warn('[Pipeline] モデル読み込み失敗—フォールバック:', error);
      this.modelReady = false;
    }}
  }}

  async process(imageSource: string): Promise<PipelineResult> {{
    await this.init();

    let placement: PlacementResult;
    let detectedLabels: string[] = [];
    let occlusionMask: {{ width: number; height: number; data: Uint8Array }} | null = null;

    this.lastImageSource = imageSource;
    this.lastDetectOutput = null;
    this.lastDepthOutput = null;

    const imgInfo = await new Promise<HTMLImageElement>((resolve, reject) => {{
      const i = new Image(); i.crossOrigin = 'anonymous'; 
      i.onload = () => resolve(i); i.onerror = reject; i.src = imageSource;
    }}).catch(() => ({{ width: 800, height: 600 }}));
    const imgW = imgInfo.width;
    const imgH = imgInfo.height;

    if (this.modelReady && this.detector) {{
      try {{
        const detOutput = await this.detector(imageSource) as DetectResult[];
        
        const normalizedOutput = detOutput.map(d => ({{
          label: d.label,
          score: d.score,
          box: {{
            xmin: d.box.xmin / imgW,
            xmax: d.box.xmax / imgW,
            ymin: d.box.ymin / imgH,
            ymax: d.box.ymax / imgH
          }}
        }}));
        
        this.lastDetectOutput = normalizedOutput;
        detectedLabels = normalizedOutput.map(s => s.label.toLowerCase());

        const hasValidTarget = normalizedOutput.some(s => isTargetLabel(s.label));
        if (!hasValidTarget) throw new Error('NOT_ENOUGH_SPACE');

        let depthAtPlacement = 0.5;
        if (this.depthEstimator) {{
          try {{
            const depthResult = await this.depthEstimator(imageSource);
            const depthTensor = (depthResult as any)?.depth;
            if (depthTensor?.data) {{
              const dw = depthTensor.dims ? depthTensor.dims[2] || depthTensor.width : 518;
              const dh = depthTensor.dims ? depthTensor.dims[1] || depthTensor.height : 518;
              this.lastDepthOutput = {{ width: dw, height: dh, data: depthTensor.data as Float32Array }};
            }}
          }} catch (e) {{
            console.warn('[Pipeline] 深度推定スキップ:', e);
          }}
        }}

        placement = this.calculatePlacement(normalizedOutput, imgW, imgH);

        if (this.lastDepthOutput) {{
           depthAtPlacement = this.sampleDepthAtBounds(this.lastDepthOutput.data, this.lastDepthOutput.width, this.lastDepthOutput.height, placement);
           placement.depthValue = depthAtPlacement;
           occlusionMask = this.buildDepthOcclusionMask(this.lastDepthOutput, placement);
        }}
      }} catch (err: any) {{
        if (err.message === 'NOT_ENOUGH_SPACE') throw err;
        console.warn('[Pipeline] 推論失敗—フォールバック:', err);
        placement = this.fallbackPlacement();
      }}
    }} else {{
      placement = this.fallbackPlacement();
    }}

    const catAsset = this.pickCatAsset(placement.pose);
    this.lastCatAsset = catAsset;

    const imageDataUrl = await this.composeOverlay(imageSource, placement, occlusionMask, catAsset);
    let debugImageDataUrl = undefined;
    if (this.lastDetectOutput) {{
      debugImageDataUrl = await this.drawDebugBoxes(imageSource, this.lastDetectOutput, imgW, imgH);
    }}
    
    return {{ imageDataUrl, debugImageDataUrl, placement, detectedLabels, modelLoaded: this.modelReady }};
  }}

  async recompose(newPlacement: PlacementResult): Promise<string | null> {{
    if (!this.lastImageSource || !this.lastCatAsset) return null;
    let occlusionMask = null;
    if (this.lastDepthOutput) {{
      newPlacement.depthValue = this.sampleDepthAtBounds(this.lastDepthOutput.data, this.lastDepthOutput.width, this.lastDepthOutput.height, newPlacement);
      occlusionMask = this.buildDepthOcclusionMask(this.lastDepthOutput, newPlacement);
    }}
    return this.composeOverlay(this.lastImageSource, newPlacement, occlusionMask, this.lastCatAsset);
  }}

  private drawDebugBoxes(source: string, segments: DetectResult[], imgW: number, imgH: number): Promise<string> {{
    return new Promise((resolve) => {{
      const img = new Image();
      img.crossOrigin = 'anonymous';
      img.onload = () => {{
        const canvas = document.createElement('canvas');
        canvas.width = img.width;
        canvas.height = img.height;
        const ctx = canvas.getContext('2d')!;
        ctx.drawImage(img, 0, 0);

        for (const seg of segments) {{
          if (seg.label === 'unlabeled') continue;
          
          const textX = (seg.box.xmin + seg.box.xmax) / 2 * img.width;
          const textY = (seg.box.ymin + seg.box.ymax) / 2 * img.height;
          
          ctx.strokeStyle = isTargetLabel(seg.label) ? 'rgba(52, 199, 89, 0.8)' : 'rgba(255, 59, 48, 0.5)';
          ctx.lineWidth = 4;
          ctx.strokeRect(seg.box.xmin * img.width, seg.box.ymin * img.height, (seg.box.xmax - seg.box.xmin) * img.width, (seg.box.ymax - seg.box.ymin) * img.height);
          
          ctx.font = 'bold 24px sans-serif';
          ctx.textAlign = 'center';
          ctx.textBaseline = 'middle';
          ctx.lineWidth = 4;
          ctx.strokeStyle = 'rgba(0,0,0,0.8)';
          ctx.strokeText(seg.label + ' ' + Math.round(seg.score*100) + '%', textX, textY);
          ctx.fillStyle = 'white';
          ctx.fillText(seg.label + ' ' + Math.round(seg.score*100) + '%', textX, textY);
        }}
        resolve(canvas.toDataURL('image/jpeg', 0.8));
      }};
      img.onerror = () => resolve(source);
      img.src = source;
    }});
  }}

  private calculatePlacement(segments: DetectResult[], imgW: number, imgH: number): PlacementResult {{
    const surfaceTargets = segments.filter(s => isTargetLabel(s.label) && getBehavior(s.label).position === 'surface' && s.box.ymax > 0.4);
    const candidates = surfaceTargets.length > 0 ? surfaceTargets : segments.filter(s => isTargetLabel(s.label));

    if (candidates.length === 0) return this.fallbackPlacement();

    const target = candidates[Math.floor(Math.random() * candidates.length)];
    const bounds = target.box;
    const behavior = getBehavior(target.label);
    const objW = bounds.xmax - bounds.xmin;
    const objH = bounds.ymax - bounds.ymin;

    let x: number, y: number;
    if (behavior.position === 'behind') {{
      const isRight = Math.random() > 0.5;
      x = isRight ? bounds.xmax + 0.015 : bounds.xmin - 0.015;
      y = bounds.ymin + objH * (0.15 + (Math.random() - 0.5) * 0.1);
    }} else if (behavior.position === 'beside') {{
      const isRight = Math.random() > 0.5;
      x = isRight ? bounds.xmax + 0.05 : bounds.xmin - 0.05;
      y = (bounds.ymin + bounds.ymax) / 2 + objH * (Math.random() - 0.5) * 0.15;
    }} else {{
      x = (bounds.xmin + bounds.xmax) / 2 + objW * (Math.random() - 0.5) * 0.4;
      y = (bounds.ymin + bounds.ymax) / 2 + objH * (Math.random() - 0.5) * 0.4;
    }}

    x = Math.max(0.05, Math.min(0.90, x));
    y = Math.max(0.05, Math.min(0.92, y));

    const baseScale = Math.max(0.08, Math.min(0.22, objW * 0.65));
    const scale = baseScale;

    return {{
      x, y, scale,
      rotation: (Math.random() - 0.5) * 8,
      depthValue: 0.5,
      pose: behavior.pose,
      reason: \`\${target.label} を基準に配置\`,
      targetLabel: target.label,
    }};
  }}

  private sampleDepthAtBounds(
    depthData: Float32Array,
    w: number,
    h: number,
    placement: PlacementResult
  ): number {{
    const cx = Math.floor(placement.x * w);
    const cy = Math.floor(placement.y * h);
    const idx = cy * w + cx;
    const raw = depthData[Math.min(max(0, idx), depthData.length - 1)] ?? 0.5;
    return Math.max(0, Math.min(1, raw / 255.0));
  }}

  private buildDepthOcclusionMask(
    depthResult: {{ width: number; height: number; data: Float32Array }},
    placement: PlacementResult,
  ): {{ width: number; height: number; data: Uint8Array }} | null {{
    const {{ width, height, data }} = depthResult;
    const merged = new Uint8Array(width * height);

    const px = Math.floor(placement.x * width);
    const py = Math.floor(placement.y * height);
    if (py < 0 || py >= height || px < 0 || px >= width) return null;

    const baseDepth = data[py * width + px]; // The physical depth at cat's feet

    for (let i = 0; i < data.length; i++) {{
      // Value > baseDepth indicates closer to camera (255 is nearest)
      // We mask pixels that are significantly closer than the cat!
      if (data[i] > baseDepth + 15) {{ // small threshold to avoid auto-masking floor
        merged[i] = 255;
      }}
    }}

    return {{ width, height, data: merged }};
  }}

  private fallbackPlacement(): PlacementResult {{
    return {{
      x: 0.62, y: 0.52, scale: 0.15,
      rotation: -5, depthValue: 0.5, pose: 'peeking', reason: 'フォールバック',
    }};
  }}

  {composeOverlay}

  {drawGroundShadow}

  {applyOcclusionMask}

  {pickCatAsset}

  {drawFallbackCatLines}
}}

export const pipelineInstance = new ShirettoPipeline();
"""

with open('pipeline.ts', 'w') as f:
    f.write(full_code.replace("max(0, idx)", "Math.max(0, idx)"))
print("Updated pipeline.ts successfully")
