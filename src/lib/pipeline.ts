/**
 * Shiretto Cat: Edge AI Pipeline
 * @huggingface/transformers v3 (ESMネイティブ、Vite互換)
 */

import type { ImageSegmentationPipeline, DepthEstimationPipeline } from '@huggingface/transformers';

/** セグメンテーション結果の型 */
export interface SegmentResult {
  label: string;
  score: number;
  mask: { width: number; height: number; data: Uint8Array };
}

/** 配置計算の結果 */
export interface PlacementResult {
  x: number;
  y: number;
  scale: number;
  rotation: number;
  reason: string;
}

/** パイプラインの処理結果 */
export interface PipelineResult {
  imageDataUrl: string;
  placement: PlacementResult;
  detectedLabels: string[];
  modelLoaded: boolean;
}

/** カフェアイテムごとの振る舞い定義 */
const CAFE_BEHAVIORS: Record<string, { anchor: 'rim' | 'base'; offsetY: number }> = {
  cup:          { anchor: 'rim',  offsetY: 0.1  },
  mug:          { anchor: 'rim',  offsetY: 0.1  },
  glass:        { anchor: 'rim',  offsetY: 0.05 },
  bottle:       { anchor: 'rim',  offsetY: 0.15 },
  bowl:         { anchor: 'rim',  offsetY: 0.1  },
  plate:        { anchor: 'rim',  offsetY: 0.02 },
  table:        { anchor: 'base', offsetY: 0    },
  desk:         { anchor: 'base', offsetY: 0    },
};

/** 対象ラベルか判定する */
function isTargetLabel(label: string): boolean {
  const lower = label.toLowerCase();
  return Object.keys(CAFE_BEHAVIORS).some(key => lower.includes(key));
}

class ShirettoPipeline {
  private segmenter: ImageSegmentationPipeline | null = null;
  private depthEstimator: DepthEstimationPipeline | null = null;
  private initPromise: Promise<void> | null = null;
  private modelReady = false;

  /** モデルの初期化（遅延・一度だけ） */
  async init(): Promise<void> {
    if (this.modelReady) return;
    if (this.initPromise) return this.initPromise;

    this.initPromise = this._loadModels();
    return this.initPromise;
  }

  private async _loadModels(): Promise<void> {
    try {
      // dynamic import でモジュール読み込みのタイミングを制御する
      const { pipeline, env } = await import('@huggingface/transformers');

      env.allowLocalModels = false;
      env.useBrowserCache = true;

      console.log('[Pipeline] モデル読み込み開始...');

      this.segmenter = await pipeline(
        'image-segmentation',
        'Xenova/segformer-b0-finetuned-ade-512-512',
      ) as ImageSegmentationPipeline;

      this.depthEstimator = await pipeline(
        'depth-estimation',
        'Xenova/depth-anything-small-hf',
      ) as DepthEstimationPipeline;

      this.modelReady = true;
      console.log('[Pipeline] モデル読み込み完了');
    } catch (error) {
      console.warn('[Pipeline] モデル読み込みに失敗。フォールバックモードで動作します:', error);
      this.modelReady = false;
    }
  }

  /** メイン処理 */
  async process(imageSource: string): Promise<PipelineResult> {
    await this.init();

    let placement: PlacementResult;
    let detectedLabels: string[] = [];

    if (this.modelReady && this.segmenter) {
      try {
        const segOutput = await this.segmenter(imageSource) as SegmentResult[];
        detectedLabels = segOutput
          .map(s => s.label.toLowerCase())
          .filter(l => l !== 'unlabeled');

        placement = this.calculatePlacement(segOutput);
      } catch (err) {
        console.warn('[Pipeline] 推論失敗。フォールバック配置を使用:', err);
        placement = this.fallbackPlacement();
      }
    } else {
      placement = this.fallbackPlacement();
    }

    const imageDataUrl = await this.composeOverlay(imageSource, placement);

    return {
      imageDataUrl,
      placement,
      detectedLabels,
      modelLoaded: this.modelReady,
    };
  }

  /** セグメンテーション結果から最適な配置を計算 */
  private calculatePlacement(segments: SegmentResult[]): PlacementResult {
    // カフェアイテムを探す
    const target = segments.find(s => isTargetLabel(s.label));

    if (!target || !target.mask) {
      return this.fallbackPlacement();
    }

    const bounds = this.extractBounds(target.mask);
    const label = target.label.toLowerCase();
    const behavior = Object.entries(CAFE_BEHAVIORS).find(([k]) => label.includes(k))?.[1]
      ?? { anchor: 'base' as const, offsetY: 0 };

    let x: number, y: number;

    if (behavior.anchor === 'rim') {
      // 容器のフチの横に配置
      x = bounds.maxX + 0.02;
      y = bounds.minY + (bounds.maxY - bounds.minY) * behavior.offsetY;
    } else {
      // テーブル面に配置
      x = (bounds.minX + bounds.maxX) / 2;
      y = bounds.maxY;
    }

    // 画面外に出ないようクランプ
    x = Math.max(0.08, Math.min(0.92, x));
    y = Math.max(0.08, Math.min(0.92, y));

    const objectWidth = bounds.maxX - bounds.minX;
    const scale = Math.max(0.08, Math.min(0.22, objectWidth * 0.7));

    return {
      x,
      y,
      scale,
      rotation: (Math.random() - 0.5) * 10,
      reason: `${target.label} を検出。${behavior.anchor === 'rim' ? 'フチ横' : 'テーブル面'}に配置。`,
    };
  }

  /** フォールバック（AIなし時の固定配置） */
  private fallbackPlacement(): PlacementResult {
    return {
      x: 0.65,
      y: 0.55,
      scale: 0.15,
      rotation: -5,
      reason: 'フォールバック配置（AI未使用）',
    };
  }

  /** マスクからバウンディングボックスを抽出 */
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

  /** 猫アセットを写真の上にオーバーレイ合成 */
  private composeOverlay(source: string, placement: PlacementResult): Promise<string> {
    return new Promise((resolve) => {
      const photo = new Image();
      photo.crossOrigin = 'anonymous';

      photo.onload = () => {
        const canvas = document.createElement('canvas');
        canvas.width = photo.width;
        canvas.height = photo.height;
        const ctx = canvas.getContext('2d');
        if (!ctx) { resolve(source); return; }

        // 元画像を描画
        ctx.drawImage(photo, 0, 0);

        // 猫アセットを読み込み・合成
        const cat = new Image();
        cat.crossOrigin = 'anonymous';
        cat.src = '/peeking_cat.png';

        cat.onload = () => {
          const px = placement.x * canvas.width;
          const py = placement.y * canvas.height;
          const size = placement.scale * canvas.width;

          ctx.save();
          ctx.translate(px, py);
          ctx.rotate((placement.rotation * Math.PI) / 180);
          ctx.drawImage(cat, -size / 2, -size / 2, size, size);
          ctx.restore();

          resolve(canvas.toDataURL('image/jpeg', 0.92));
        };

        cat.onerror = () => {
          console.error('[Pipeline] 猫アセットの読み込みに失敗');
          resolve(canvas.toDataURL('image/jpeg', 0.92));
        };
      };

      photo.onerror = () => resolve(source);
      photo.src = source;
    });
  }
}

export const pipelineInstance = new ShirettoPipeline();
