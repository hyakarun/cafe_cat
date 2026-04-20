/**
 * Shiretto Cat: Edge AI Pipeline v2
 * ─────────────────────────────────
 * 3つの「自然に見せる魔法」を実装:
 *   1. Depth-based perspective scale  —— 奥ほど猫を小さく
 *   2. Ambient shadow                 —— 足元に接地影
 *   3. Occlusion masking              —— セグメントマスクで物体の陰に隠す
 */

// 型は any で扱い、@huggingface/transformers v4 の型差異を吸収する

/* ─── 型定義 ────────────────────────────────────────────────── */

export interface DetectResult {
  label: string;
  score: number;
  box: { xmin: number; ymin: number; xmax: number; ymax: number };
}

export interface PlacementResult {
  x: number;           // 0-1 正規化座標
  y: number;
  scale: number;       // canvas.width に対する割合
  rotation: number;    // deg
  depthValue: number;  // 0=手前, 1=奥（遠近感スケール補正用）
  pose: 'peeking' | 'standing' | 'walking';
  reason: string;
  targetLabel?: string; // 猫が基準とした対象物のラベル
}

export interface PipelineResult {
  imageDataUrl: string;
  debugImageDataUrl?: string; // AIの認識範囲を可視化した画像
  placement: PlacementResult;
  detectedLabels: string[];
  modelLoaded: boolean;
}

/* ─── カフェアイテム別ふるまい ──────────────────────────────── */

const CAFE_BEHAVIORS: Record<string, {
  position: 'behind' | 'beside' | 'surface';
  pose: PlacementResult['pose'];
}> = {
  // 高めの物体（後ろから覗き込む）
  cup:       { position: 'behind',  pose: 'peeking' },
  mug:       { position: 'behind',  pose: 'peeking' },
  glass:     { position: 'behind',  pose: 'peeking' },
  bottle:    { position: 'behind',  pose: 'peeking' },
  bowl:      { position: 'behind',  pose: 'peeking' },
  
  // 平たい物体（横に立つ・座る）
  plate:     { position: 'beside',  pose: 'standing' },
  spoon:     { position: 'beside',  pose: 'standing' },
  fork:      { position: 'beside',  pose: 'standing' },
  knife:     { position: 'beside',  pose: 'standing' },
  
  // テーブル面など（表面を歩く・寝転がる）
  "dining table": { position: 'surface', pose: 'walking' },
  bed:       { position: 'surface', pose: 'walking' },
  couch:     { position: 'surface', pose: 'walking' },
  chair:     { position: 'surface', pose: 'walking' },
};

function getBehavior(label: string) {
  const lower = label.toLowerCase();
  const match = Object.entries(CAFE_BEHAVIORS).find(([k]) => lower.includes(k));
  return match?.[1] ?? { position: 'surface' as const, pose: 'walking' as const };
}

function isTargetLabel(label: string): boolean {
  const lower = label.toLowerCase();
  return Object.keys(CAFE_BEHAVIORS).some(k => lower.includes(k));
}

/* ─── パイプライン本体 ──────────────────────────────────────── */

class ShirettoPipeline {
  private detector: any = null;
  private depthEstimator: any = null;
  private initPromise: Promise<void> | null = null;
  private modelReady = false;

  // キャッシュ（再合成用）
  private lastImageSource: string | null = null;
  private lastDetectOutput: DetectResult[] | null = null;
  private lastDepthOutput: { width: number; height: number; channels: number; data: any } | null = null;
  private lastCatAsset: string | null = null;

  async init(): Promise<void> {
    if (this.modelReady) return;
    if (this.initPromise) return this.initPromise;
    this.initPromise = this._loadModels();
    return this.initPromise;
  }

  private async _loadModels(): Promise<void> {
    try {
      const { pipeline, env } = await import('@huggingface/transformers');
      env.allowLocalModels = false;
      env.useBrowserCache = true;

      console.log('[Pipeline] モデル読み込み開始...');

      // Gemma 4 E2Bモデル（超高精度なマルチモーダルによる物体検出）
      // ※WebGPUなどのサポート状況に応じて自動で最適なランタイムが選ばれます
      this.detector = await pipeline(
        'image-text-to-text',
        'google/gemma-4-e2b',
        { device: 'webgpu' } // デバイスのGPUを優先使用
      ).catch(async e => {
        console.warn('WebGPUの初期化に失敗しました。CPU・WASMモードでフォールバックします', e);
        return await pipeline('image-text-to-text', 'google/gemma-4-e2b');
      });

      // 深度推定（失敗してもセグメンテーションだけで動作継続）
      try {
        this.depthEstimator = await pipeline(
          'depth-estimation',
          'Xenova/depth-anything-small-hf',
        );
        console.log('[Pipeline] 深度推定モデルも読み込み完了 ✓');
      } catch (depthErr) {
        console.warn('[Pipeline] 深度推定モデルは省略:', depthErr);
      }

      this.modelReady = true;
      console.log('[Pipeline] モデル読み込み完了 ✓');
    } catch (error) {
      console.warn('[Pipeline] モデル読み込み失敗—フォールバックで動作:', error);
      this.modelReady = false;
    }
  }

  /* ── メイン処理 ─────────────────────────────────────────── */

  async process(imageSource: string): Promise<PipelineResult> {
    await this.init();

    let placement: PlacementResult;
    let detectedLabels: string[] = [];
    let occlusionMask: { width: number; height: number; data: Uint8Array } | null = null;

    this.lastImageSource = imageSource;
    this.lastDetectOutput = null; // リセット
    this.lastDepthOutput = null;

    // 元画像のサイズを取得（正規化座標に変換するため）
    const imgInfo = await new Promise<HTMLImageElement>((resolve, reject) => {
      const i = new Image(); i.crossOrigin = 'anonymous'; 
      i.onload = () => resolve(i); i.onerror = reject; i.src = imageSource;
    }).catch(() => ({ width: 800, height: 600 }));
    const imgW = imgInfo.width;
    const imgH = imgInfo.height;

    if (this.modelReady && this.detector) {
      try {
        // Gemma 4に画像の物体検出と座標出力を依頼
        const prompt = "Extract bounding boxes for all cafe items (e.g. dining table, cup, plate, cake, fork, spoon, knife, bottle). Return ONLY a JSON array formatted as [{ \"label\": \"item name\", \"box_2d\": [ymin, xmin, ymax, xmax] }] where coordinates are 0-1000.";
        const gemmaResult = await this.detector(imageSource, prompt, { max_new_tokens: 512 });
        const textOut: string = gemmaResult[0]?.generated_text || "[]";
        
        // 余分な文章を省いてJSONだけを抽出
        const jsonMatch = textOut.match(/\[.*\]/s);
        let parsedBoxes: any[] = [];
        if (jsonMatch) {
          try {
            parsedBoxes = JSON.parse(jsonMatch[0]);
          } catch (e) {
            console.warn("Gemma JSON Parse Error:", e);
          }
        }
        
        // bounding boxの座標（0-1000）を0-1に正規化
        const normalizedOutput = parsedBoxes.map(d => ({
          label: d.label.toLowerCase(),
          score: 0.99, // Gemmaは確信度を出さないため固定
          box: {
            xmin: d.box_2d[1] / 1000,
            ymin: d.box_2d[0] / 1000,
            xmax: d.box_2d[3] / 1000,
            ymax: d.box_2d[2] / 1000
          }
        }));
        
        this.lastDetectOutput = normalizedOutput;
        detectedLabels = normalizedOutput
          .map(s => s.label.toLowerCase())
          .sort((a,b) => a.localeCompare(b)); // Sort just for uniqueness if needed, keeping simple

        const hasValidTarget = normalizedOutput.some(s => isTargetLabel(s.label));
        if (!hasValidTarget) {
          throw new Error('NOT_ENOUGH_SPACE');
        }

        // ── 深度推定で奥行きマップ全体を取得 ──
        let depthAtPlacement = 0.5;
        if (this.depthEstimator) {
          try {
            const depthResult = await this.depthEstimator(imageSource);
            const depthTensor = (depthResult as any)?.depth;
            if (depthTensor?.data) {
              const dw = depthTensor.width || 518;
              const dh = depthTensor.height || 518;
              const channels = depthTensor.channels || 1;
              this.lastDepthOutput = { width: dw, height: dh, channels, data: depthTensor.data };
            }
          } catch (e) {
            console.warn('[Pipeline] 深度推定スキップ:', e);
          }
        }

        placement = this.calculatePlacement(normalizedOutput);

        if (this.lastDepthOutput) {
          depthAtPlacement = this.sampleDepthAtBounds(
            this.lastDepthOutput,
            placement
          );
          placement.depthValue = depthAtPlacement;
          
          // ── 遮蔽マスク：深度マップを使って猫の後ろにある物体を隠すマスクの生成 ──
          occlusionMask = this.buildDepthOcclusionMask(this.lastDepthOutput, placement);
        }

      } catch (err: any) {
        if (err.message === 'NOT_ENOUGH_SPACE') {
          throw err;
        }
        console.warn('[Pipeline] 推論失敗—フォールバック配置:', err);
        placement = this.fallbackPlacement();
      }
    } else {
      placement = this.fallbackPlacement();
    }

    const catAsset = this.pickCatAsset(placement.pose);
    this.lastCatAsset = catAsset;

    const imageDataUrl = await this.composeOverlay(imageSource, placement, occlusionMask, catAsset);
    let debugImageDataUrl = undefined;
    if (this.lastDetectOutput) {
      debugImageDataUrl = await this.drawDebugMasks(imageSource, this.lastDetectOutput, imgW, imgH);
    }
    
    return { imageDataUrl, debugImageDataUrl, placement, detectedLabels, modelLoaded: this.modelReady };
  }

  async recompose(newPlacement: PlacementResult): Promise<string | null> {
    if (!this.lastImageSource || !this.lastCatAsset) return null;
    
    let occlusionMask = null;
    if (this.lastDepthOutput) {
      newPlacement.depthValue = this.sampleDepthAtBounds(
        this.lastDepthOutput,
        newPlacement
      );
      occlusionMask = this.buildDepthOcclusionMask(this.lastDepthOutput, newPlacement);
    }
    return this.composeOverlay(this.lastImageSource, newPlacement, occlusionMask, this.lastCatAsset);
  }

  private drawDebugMasks(source: string, segments: DetectResult[], imgW: number, imgH: number): Promise<string> {
    return new Promise((resolve) => {
      const img = new Image();
      img.crossOrigin = 'anonymous';
      img.onload = () => {
        const canvas = document.createElement('canvas');
        canvas.width = img.width;
        canvas.height = img.height;
        const ctx = canvas.getContext('2d')!;
        ctx.drawImage(img, 0, 0);

        for (const seg of segments) {
          if (seg.label === 'unlabeled') continue;
          
          const bx = seg.box.xmin * img.width;
          const by = seg.box.ymin * img.height;
          const bw = (seg.box.xmax - seg.box.xmin) * img.width;
          const bh = (seg.box.ymax - seg.box.ymin) * img.height;

          const textX = bx + bw / 2;
          const textY = by + bh / 2;
          
          ctx.strokeStyle = isTargetLabel(seg.label) ? 'rgba(52, 199, 89, 0.8)' : 'rgba(255, 59, 48, 0.5)';
          ctx.lineWidth = 4;
          ctx.strokeRect(bx, by, bw, bh);

          ctx.font = 'bold 24px sans-serif';
          ctx.textAlign = 'center';
          ctx.textBaseline = 'middle';
          ctx.lineWidth = 4;
          ctx.strokeStyle = 'rgba(0,0,0,0.8)';
          const text = `${seg.label} (${Math.round(seg.score * 100)}%)`;
          ctx.strokeText(text, textX, textY);
          ctx.fillStyle = 'white';
          ctx.fillText(text, textX, textY);
        }

        resolve(canvas.toDataURL('image/jpeg', 0.8));
      };
      img.onerror = () => resolve(source);
      img.src = source;
    });
  }

  private calculatePlacement(segments: DetectResult[]): PlacementResult {
    // 前景（下 40% 以下）にある表面（テーブルやデスク）だけを候補にする
    const surfaceTargets = segments.filter(s => {
      if (!isTargetLabel(s.label)) return false;
      const behavior = getBehavior(s.label);
      if (behavior.position !== 'surface') return false;
      return s.box.ymax > 0.4;
    });

    const candidates = surfaceTargets.length > 0 
      ? surfaceTargets 
      : segments.filter(s => isTargetLabel(s.label) && getBehavior(s.label).position === 'surface');

    if (candidates.length === 0) return this.fallbackPlacement();

    const target = candidates[Math.floor(Math.random() * candidates.length)];
    const bounds = target.box;
    const behavior = getBehavior(target.label);
    const objW = bounds.xmax - bounds.xmin;
    const objH = bounds.ymax - bounds.ymin;

    let x: number, y: number;
    if (behavior.position === 'behind') {
      const isRight = Math.random() > 0.5;
      x = isRight ? bounds.xmax + 0.015 : bounds.xmin - 0.015;
      const yJitter = (Math.random() - 0.5) * 0.1;
      y = bounds.ymin + objH * (0.15 + yJitter);
    } else if (behavior.position === 'beside') {
      const isRight = Math.random() > 0.5;
      x = isRight ? bounds.xmax + 0.05 : bounds.xmin - 0.05; 
      const yJitter = (Math.random() - 0.5) * 0.15;
      y = (bounds.ymin + bounds.ymax) / 2 + objH * yJitter;
    } else {
      // テーブルなど平面の上（枠内）に配置する。ただし他の物体（コップや皿など）のボックスを避ける
      const obstacles = segments.filter(s => getBehavior(s.label).position !== 'surface');
      let found = false;
      
      // 最大50回ランダムな座標を試し、空いている場所を探す
      for (let attempt = 0; attempt < 50; attempt++) {
        const testX = bounds.xmin + Math.random() * objW;
        const testY = bounds.ymin + Math.random() * objH;
        
        // 他の物体のボックスに被っていないかチェック（猫の体サイズ分のマージンを取る）
        const hit = obstacles.some(obs => {
           const marginX = 0.04;
           const marginY = 0.04;
           return (
             testX > obs.box.xmin - marginX && 
             testX < obs.box.xmax + marginX &&
             testY > obs.box.ymin - marginY &&
             testY < obs.box.ymax + marginY
           );
        });

        if (!hit) {
          x = testX;
          y = testY;
          found = true;
          break;
        }
      }

      // もし50回試してダメなら、元のランダム中央ロジックにフォールバック
      if (!found) {
        x = (bounds.xmin + bounds.xmax) / 2 + objW * (Math.random() - 0.5) * 0.4;
        y = (bounds.ymin + bounds.ymax) / 2 + objH * (Math.random() - 0.5) * 0.4;
      }
    }

    x = Math.max(0.05, Math.min(0.90, x));
    y = Math.max(0.05, Math.min(0.92, y));

    const baseScale = Math.max(0.08, Math.min(0.22, objW * 0.65));
    const scale = baseScale; // Depth scale will be applied later when we sample depth directly

    return {
      x, y, scale,
      rotation: (Math.random() - 0.5) * 8,
      depthValue: 0.5,
      pose: behavior.pose,
      reason: `${target.label} を基準に配置`,
      targetLabel: target.label,
    };
  }

  private sampleDepthAtBounds(
    depthResult: { width: number; height: number; channels: number; data: any },
    placement: PlacementResult
  ): number {
    const { width: w, height: h, channels, data } = depthResult;
    const cx = Math.max(0, Math.min(w - 1, Math.floor(placement.x * w)));
    const cy = Math.max(0, Math.min(h - 1, Math.floor(placement.y * h)));
    const idx = (cy * w + cx) * channels;
    const raw = data[Math.max(0, Math.min(idx, data.length - 1))] ?? 128;
    return Math.max(0, Math.min(1, raw / 255.0));
  }

  private buildDepthOcclusionMask(
    depthResult: { width: number; height: number; channels: number; data: any },
    placement: PlacementResult,
  ): { width: number; height: number; data: Uint8Array } | null {
    const { width, height, channels, data } = depthResult;
    const merged = new Uint8Array(width * height);

    const px = Math.max(0, Math.min(width - 1, Math.floor(placement.x * width)));
    const py = Math.max(0, Math.min(height - 1, Math.floor(placement.y * height)));
    const baseIdx = (py * width + px) * channels;
    const baseDepthValue = data[baseIdx];

    const DEPTH_THRESHOLD = 5;

    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const i = y * width + x;
        const dIdx = i * channels;
        if (data[dIdx] > baseDepthValue + DEPTH_THRESHOLD) {
          merged[i] = 255;
        }
      }
    }

    return { width, height, data: merged };
  }

  private fallbackPlacement(): PlacementResult {
    return {
      x: 0.62, y: 0.52, scale: 0.15,
      rotation: -5, depthValue: 0.5,
      pose: 'peeking',
      reason: 'フォールバック配置（AI未使用）',
    };
  }


  /* ── 合成（3つの魔法を適用） ────────────────────────────── */

  private composeOverlay(
    source: string,
    placement: PlacementResult,
    occlusionMask: { width: number; height: number; data: Uint8Array } | null,
    catAssetUrl: string
  ): Promise<string> {
    return new Promise((resolve) => {
      const photo = new Image();
      photo.crossOrigin = 'anonymous';

      photo.onload = () => {
        const W = photo.width;
        const H = photo.height;
        const canvas = document.createElement('canvas');
        canvas.width = W;
        canvas.height = H;
        const ctx = canvas.getContext('2d')!;
        ctx.drawImage(photo, 0, 0);

        const px = placement.x * W;
        const py = placement.y * H;
        const size = placement.scale * W;

        // ── 魔法 2: 接地影 ────────────────────────────────
        this.drawGroundShadow(ctx, px, py, size, placement.depthValue);

        // 猫アセットを描画してから遮蔽マスクを上から適用
        const catCanvas = document.createElement('canvas');
        catCanvas.width = W;
        catCanvas.height = H;
        const catCtx = catCanvas.getContext('2d')!;

        const cat = new Image();
        cat.crossOrigin = 'anonymous';
        cat.src = catAssetUrl;

        cat.onload = () => {
          catCtx.save();
          
          // 白い線画が明るい背景でも見えるように、黒いドロップシャドウをかける
          catCtx.filter = 'drop-shadow(0px 2px 5px rgba(0, 0, 0, 0.45))';
          catCtx.globalCompositeOperation = 'source-over';
          
          catCtx.translate(px, py);
          catCtx.rotate((placement.rotation * Math.PI) / 180);
          catCtx.drawImage(cat, -size / 2, -size / 2, size, size);
          catCtx.restore();

          // ── 魔法 3: 遮蔽マスクで一部を消す ─────────────
          if (occlusionMask) {
            this.applyOcclusionMask(catCtx, occlusionMask, W, H);
          }

          // 合成済み猫レイヤーを写真に重ねる
          ctx.drawImage(catCanvas, 0, 0);
          resolve(canvas.toDataURL('image/jpeg', 0.92));
        };

        cat.onerror = () => {
          // 猫アセットが読み込めなかった場合はCanvas線画で代替
          ctx.save();
          ctx.translate(px, py);
          ctx.rotate((placement.rotation * Math.PI) / 180);
          this.drawFallbackCatLines(ctx, size, placement.pose);
          ctx.restore();
          resolve(canvas.toDataURL('image/jpeg', 0.92));
        };
      };

      photo.onerror = () => resolve(source);
      photo.src = source;
    });
  }

  /** 魔法 2: 楕円形の接地影 */
  private drawGroundShadow(
    ctx: CanvasRenderingContext2D,
    px: number,
    py: number,
    size: number,
    depthValue: number,
  ) {
    const shadowOpacity = 0.18 - depthValue * 0.10; // 手前ほど濃い影
    const rx = size * 0.38;
    const ry = size * 0.10;

    ctx.save();
    ctx.translate(px, py + size * 0.05);
    const grad = ctx.createRadialGradient(0, 0, 0, 0, 0, rx);
    grad.addColorStop(0, `rgba(0,0,0,${shadowOpacity})`);
    grad.addColorStop(1, 'rgba(0,0,0,0)');
    ctx.scale(1, ry / rx);
    ctx.beginPath();
    ctx.arc(0, 0, rx, 0, Math.PI * 2);
    ctx.fillStyle = grad;
    ctx.fill();
    ctx.restore();
  }

  /** 魔法 3: 遮蔽マスクを適用（物体の前にある部分を消す）
   *  canvas の destination-out モードでマスク領域を透明化。
   */
  private applyOcclusionMask(
    catCtx: CanvasRenderingContext2D,
    occlusionMask: { width: number; height: number; data: Uint8Array },
    W: number,
    H: number,
  ) {
    // マスク用の一時 canvas
    const maskCanvas = document.createElement('canvas');
    maskCanvas.width = W;
    maskCanvas.height = H;
    const maskCtx = maskCanvas.getContext('2d')!;

    const imgData = maskCtx.createImageData(W, H);
    const scaleX = W / occlusionMask.width;
    const scaleY = H / occlusionMask.height;

    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        const sx = Math.floor(x / scaleX);
        const sy = Math.floor(y / scaleY);
        const srcIdx = sy * occlusionMask.width + sx;
        const val = occlusionMask.data[srcIdx] ?? 0;
        const dstIdx = (y * W + x) * 4;
        imgData.data[dstIdx]     = 0;
        imgData.data[dstIdx + 1] = 0;
        imgData.data[dstIdx + 2] = 0;
        imgData.data[dstIdx + 3] = val; // マスク値 → alpha
      }
    }
    maskCtx.putImageData(imgData, 0, 0);

    // cat canvas の対象領域を destination-out で消す
    catCtx.globalCompositeOperation = 'destination-out';
    catCtx.drawImage(maskCanvas, 0, 0);
    catCtx.globalCompositeOperation = 'source-over';
  }

  private pickCatAsset(pose: string): string {
    const POSE_MAP: Record<string, string[]> = {
      peeking:  ['座り', '座り2', '座り3', '座り4', '座り5', '座り6', '座り7', '座り8'],
      standing: ['佇む', '佇む2', '佇む3', '佇む4', '佇む5'],
      lying:    ['寝転がり', '寝転がり2', '寝転がり3', '寝転がり4', '寝転がり5', '寝転がり6', '寝転がり7', '寝転がり8'],
      // 「歩き1」画像削除への対応：テーブル面(walking)は「佇む」か「寝転がり」から選ばれるようにする
      walking:  [
        '佇む', '佇む2', '佇む3', '佇む4', '佇む5',
        '寝転がり', '寝転がり2', '寝転がり3', '寝転がり4', '寝転がり5', '寝転がり6', '寝転がり7', '寝転がり8'
      ],
    };

    // 該当ポーズがなければ全カテゴリから選ぶ
    const candidates =
      POSE_MAP[pose] ??
      Object.values(POSE_MAP).flat();

    const name = candidates[Math.floor(Math.random() * candidates.length)];
    return `/${encodeURIComponent(name)}.png`;
  }

  /** 猫アセットが読み込めない場合のフォールバック線画 */
  private drawFallbackCatLines(
    ctx: CanvasRenderingContext2D,
    size: number,
    pose: PlacementResult['pose'],
  ) {
    ctx.strokeStyle = 'white';
    ctx.lineWidth = Math.max(1.5, size / 30);
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.shadowBlur = 6;
    ctx.shadowColor = 'rgba(255,255,255,0.5)';

    if (pose === 'peeking') {
      // 顔輪郭
      ctx.beginPath();
      ctx.moveTo(-size * 0.4, 0);
      ctx.quadraticCurveTo(-size * 0.4, -size * 0.4, -size * 0.25, -size * 0.5);
      ctx.lineTo(-size * 0.30, -size * 0.70);
      ctx.lineTo(-size * 0.10, -size * 0.60);
      ctx.quadraticCurveTo(0, -size * 0.65, size * 0.10, -size * 0.60);
      ctx.lineTo(size * 0.30, -size * 0.70);
      ctx.lineTo(size * 0.25, -size * 0.50);
      ctx.quadraticCurveTo(size * 0.4, -size * 0.4, size * 0.4, 0);
      // 肉球
      ctx.moveTo(-size * 0.45, 0);
      ctx.quadraticCurveTo(-size * 0.45, size * 0.06, -size * 0.25, size * 0.06);
      ctx.quadraticCurveTo(-size * 0.25, size * 0.06, -size * 0.25, 0);
      ctx.moveTo(size * 0.25, 0);
      ctx.quadraticCurveTo(size * 0.25, size * 0.06, size * 0.45, size * 0.06);
      ctx.quadraticCurveTo(size * 0.45, size * 0.06, size * 0.45, 0);
      ctx.stroke();
      // 目
      ctx.beginPath();
      ctx.arc(-size * 0.12, -size * 0.34, size * 0.025, 0, Math.PI * 2);
      ctx.arc(size * 0.12, -size * 0.34, size * 0.025, 0, Math.PI * 2);
      ctx.fillStyle = 'white';
      ctx.fill();
    } else if (pose === 'walking') {
      ctx.beginPath();
      ctx.moveTo(-size * 0.8, -size * 0.3);
      ctx.quadraticCurveTo(-size * 1.0, -size * 0.1, -size * 0.7, -size * 0.15);
      ctx.quadraticCurveTo(-size * 0.4, -size * 0.5, -size * 0.2, -size * 0.5);
      ctx.quadraticCurveTo(size * 0.1, -size * 0.6, size * 0.3, -size * 0.9);
      ctx.lineTo(size * 0.35, -size * 1.0);
      ctx.lineTo(size * 0.45, -size * 0.9);
      ctx.quadraticCurveTo(size * 0.55, -size * 0.65, size * 0.45, -size * 0.45);
      ctx.stroke();
    } else {
      ctx.beginPath();
      ctx.moveTo(-size * 0.8, -size * 0.2);
      ctx.quadraticCurveTo(-size * 1.2, 0, -size * 0.8, size * 0.2);
      ctx.quadraticCurveTo(-size * 0.5, -size * 0.2, -size * 0.3, -size * 0.6);
      ctx.quadraticCurveTo(0, -size * 0.8, size * 0.2, -size * 1.0);
      ctx.lineTo(size * 0.25, -size * 1.2);
      ctx.lineTo(size * 0.35, -size * 1.0);
      ctx.quadraticCurveTo(size * 0.5, -size * 0.8, size * 0.45, -size * 0.4);
      ctx.stroke();
    }
  }
}

export const pipelineInstance = new ShirettoPipeline();
