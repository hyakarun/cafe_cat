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

export interface SegmentResult {
  label: string;
  score: number;
  mask: { width: number; height: number; data: Uint8Array };
}

export interface PlacementResult {
  x: number;           // 0-1 正規化座標
  y: number;
  scale: number;       // canvas.width に対する割合
  rotation: number;    // deg
  depthValue: number;  // 0=手前, 1=奥（遠近感スケール補正用）
  pose: 'peeking' | 'standing' | 'walking';
  reason: string;
}

export interface PipelineResult {
  imageDataUrl: string;
  placement: PlacementResult;
  detectedLabels: string[];
  modelLoaded: boolean;
}

/* ─── カフェアイテム別ふるまい ──────────────────────────────── */

const CAFE_BEHAVIORS: Record<string, {
  anchor: 'rim' | 'base';
  pose: PlacementResult['pose'];
  offsetY: number;
}> = {
  cup:    { anchor: 'rim',  pose: 'peeking',  offsetY: 0.10 },
  mug:    { anchor: 'rim',  pose: 'peeking',  offsetY: 0.10 },
  glass:  { anchor: 'rim',  pose: 'peeking',  offsetY: 0.05 },
  bottle: { anchor: 'rim',  pose: 'peeking',  offsetY: 0.15 },
  bowl:   { anchor: 'rim',  pose: 'peeking',  offsetY: 0.10 },
  plate:  { anchor: 'rim',  pose: 'standing', offsetY: 0.00 },
  table:  { anchor: 'base', pose: 'walking',  offsetY: 0.00 },
  desk:   { anchor: 'base', pose: 'walking',  offsetY: 0.00 },
};

function getBehavior(label: string) {
  const lower = label.toLowerCase();
  const match = Object.entries(CAFE_BEHAVIORS).find(([k]) => lower.includes(k));
  return match?.[1] ?? { anchor: 'base' as const, pose: 'walking' as const, offsetY: 0 };
}

function isTargetLabel(label: string): boolean {
  const lower = label.toLowerCase();
  return Object.keys(CAFE_BEHAVIORS).some(k => lower.includes(k));
}

/* ─── パイプライン本体 ──────────────────────────────────────── */

class ShirettoPipeline {
  private segmenter: any = null;
  private depthEstimator: any = null;
  private initPromise: Promise<void> | null = null;
  private modelReady = false;

  // キャッシュ（再合成用）
  private lastImageSource: string | null = null;
  private lastSegOutput: SegmentResult[] | null = null;
  private lastCatAsset: string | null = null;

  async init(): Promise<void> {
    if (this.modelReady) return;
    if (this.initPromise) return this.initPromise;
    this.initPromise = this._loadModels();
    return this.initPromise;
  }

  private async _loadModels(): Promise<void> {
    try {
      const { pipeline, env } = await import('@xenova/transformers');
      env.allowLocalModels = false;
      env.useBrowserCache = true;

      console.log('[Pipeline] モデル読み込み開始...');

      // セグメンテーション — v4 で動作確認済みモデル
      this.segmenter = await pipeline(
        'image-segmentation',
        'Xenova/detr-resnet-50-panoptic',
      );

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
    this.lastSegOutput = null; // リセット

    if (this.modelReady && this.segmenter) {
      try {
        const segOutput = await this.segmenter(imageSource) as SegmentResult[];
        this.lastSegOutput = segOutput;
        detectedLabels = segOutput
          .map(s => s.label.toLowerCase())
          .filter(l => l !== 'unlabeled');

        // ── 深度推定で奥行き値を取得 ──
        let depthAtPlacement = 0.5;
        if (this.depthEstimator) {
          try {
            // @ts-ignore HF Transformers v3 の型が不安定なため
            const depthResult = await this.depthEstimator(imageSource);
            const depthMap = (depthResult as any)?.depth?.data as Float32Array | undefined;
            if (depthMap) {
              // ターゲット物体の中心付近の深度値を取得
              const target = segOutput.find(s => isTargetLabel(s.label));
              if (target) {
                depthAtPlacement = this.sampleDepthAtBounds(
                  depthMap,
                  target.mask.width,
                  target.mask.height,
                  this.extractBounds(target.mask),
                );
              }
            }
          } catch (e) {
            console.warn('[Pipeline] 深度推定スキップ:', e);
          }
        }

        placement = this.calculatePlacement(segOutput, depthAtPlacement);

        // ── 遮蔽マスク：猫の後ろにある物体マスクを収集 ──
        occlusionMask = this.buildOcclusionMask(segOutput, placement);
      } catch (err) {
        console.warn('[Pipeline] 推論失敗—フォールバック配置:', err);
        placement = this.fallbackPlacement();
      }
    } else {
      placement = this.fallbackPlacement();
    }

    const catAsset = this.pickCatAsset(placement.pose);
    this.lastCatAsset = catAsset;

    const imageDataUrl = await this.composeOverlay(imageSource, placement, occlusionMask, catAsset);
    return { imageDataUrl, placement, detectedLabels, modelLoaded: this.modelReady };
  }

  /** 位置調整などでの再合成（AI推論をスキップ） */
  async recompose(newPlacement: PlacementResult): Promise<string | null> {
    if (!this.lastImageSource || !this.lastCatAsset) return null;
    
    let occlusionMask = null;
    if (this.lastSegOutput) {
      occlusionMask = this.buildOcclusionMask(this.lastSegOutput, newPlacement);
    }
    return this.composeOverlay(this.lastImageSource, newPlacement, occlusionMask, this.lastCatAsset);
  }


  /* ── 配置計算 ───────────────────────────────────────────── */

  private calculatePlacement(segments: SegmentResult[], depthValue: number): PlacementResult {
    // 前景（下 40% 以下）にあるカフェアイテムを優先
    const foregroundTargets = segments.filter(s => {
      if (!isTargetLabel(s.label)) return false;
      const b = this.extractBounds(s.mask);
      return b.maxY > 0.4;
    });

    const target = foregroundTargets[0] ?? segments.find(s => isTargetLabel(s.label));
    if (!target?.mask) return this.fallbackPlacement();

    const bounds = this.extractBounds(target.mask);
    const behavior = getBehavior(target.label);
    const objW = bounds.maxX - bounds.minX;
    const objH = bounds.maxY - bounds.minY;

    let x: number, y: number;
    if (behavior.anchor === 'rim') {
      // フチの右側・少し上に出現
      x = bounds.maxX + 0.015;
      y = bounds.minY + objH * behavior.offsetY;
    } else {
      x = (bounds.minX + bounds.maxX) / 2 + objW * 0.15;
      y = bounds.maxY;
    }

    x = Math.max(0.05, Math.min(0.90, x));
    y = Math.max(0.05, Math.min(0.92, y));

    // 深度に応じてスケール補正（奥 → 小さく、手前 → 大きく）
    const depthScale = 1.0 - depthValue * 0.45;
    const baseScale = Math.max(0.08, Math.min(0.22, objW * 0.65));
    const scale = baseScale * depthScale;

    return {
      x, y, scale,
      rotation: (Math.random() - 0.5) * 8,
      depthValue,
      pose: behavior.pose,
      reason: `${target.label} を検出。depth=${depthValue.toFixed(2)}、scale=${scale.toFixed(3)}`,
    };
  }

  /** 配置場所の深度値をサンプリング */
  private sampleDepthAtBounds(
    depthMap: Float32Array,
    w: number,
    h: number,
    bounds: { minX: number; maxX: number; minY: number; maxY: number },
  ): number {
    const cx = Math.floor(((bounds.minX + bounds.maxX) / 2) * w);
    const cy = Math.floor(((bounds.minY + bounds.maxY) / 2) * h);
    const idx = cy * w + cx;
    const raw = depthMap[Math.min(idx, depthMap.length - 1)] ?? 0.5;
    return Math.max(0, Math.min(1, raw));
  }

  /** 遮蔽マスク：猫の座標に重なる物体マスクを統合 */
  private buildOcclusionMask(
    segments: SegmentResult[],
    placement: PlacementResult,
  ): { width: number; height: number; data: Uint8Array } | null {
    if (!segments.length) return null;
    const { width, height, data } = segments[0].mask;

    // 配置座標に対応するピクセル位置
    const px = Math.floor(placement.x * width);
    const py = Math.floor(placement.y * height);
    const radius = Math.floor(placement.scale * width * 0.5);

    const merged = new Uint8Array(width * height);

    for (const seg of segments) {
      if (!isTargetLabel(seg.label)) continue;
      // 猫の位置と重なるセグメントだけを遮蔽マスクに追加
      for (let i = 0; i < seg.mask.data.length; i++) {
        if (seg.mask.data[i] > 128) {
          const sx = i % width;
          const sy = Math.floor(i / width);
          const dist = Math.sqrt((sx - px) ** 2 + (sy - py) ** 2);
          if (dist < radius * 1.2) {
            merged[i] = 255;
          }
        }
      }
    }

    return { width, height, data: merged };
  }

  /** フォールバック配置 */
  private fallbackPlacement(): PlacementResult {
    return {
      x: 0.62, y: 0.52, scale: 0.15,
      rotation: -5, depthValue: 0.5,
      pose: 'peeking',
      reason: 'フォールバック配置（AI未使用）',
    };
  }

  /* ── バウンディングボックス抽出 ─────────────────────────── */

  private extractBounds(mask: { width: number; height: number; data: Uint8Array }) {
    let minX = mask.width, maxX = 0, minY = mask.height, maxY = 0;
    for (let i = 0; i < mask.data.length; i++) {
      if (mask.data[i] > 128) {
        const x = i % mask.width;
        const y = Math.floor(i / mask.width);
        if (x < minX) minX = x;
        if (x > maxX) maxX = x;
        if (y < minY) minY = y;
        if (y > maxY) maxY = y;
      }
    }
    return {
      minX: minX / mask.width,
      maxX: maxX / mask.width,
      minY: minY / mask.height,
      maxY: maxY / mask.height,
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
          catCtx.globalCompositeOperation = 'screen'; // 白線画を自然に合成
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

  /** AIのポーズ判定に応じて対応するカテゴリの画像をランダムに返す */
  private pickCatAsset(pose: string): string {
    const POSE_MAP: Record<string, string[]> = {
      peeking:  ['座り', '座り2', '座り3', '座り4', '座り5', '座り6', '座り7', '座り8'],
      standing: ['佇む', '佇む2', '佇む3', '佇む4', '佇む5'],
      walking:  ['歩き1'],
      // テーブル面・フォールバックは寝転がりも候補に
      lying:    ['寝転がり', '寝転がり2', '寝転がり3', '寝転がり4', '寝転がり5', '寝転がり6', '寝転がり7', '寝転がり8'],
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
