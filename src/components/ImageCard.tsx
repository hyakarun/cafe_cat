import React, { useState, useEffect, useRef } from 'react';
import { Share2, RotateCcw, Sparkles, SlidersHorizontal, Check } from 'lucide-react';
import { motion } from 'framer-motion';
import type { PlacementResult } from '../lib/pipeline';

interface ImageCardProps {
  image: string;
  processedImage: string | null;
  isProcessing: boolean;
  onReset: () => void;
  placement?: PlacementResult | null;
  onAdjust?: (newPlacement: PlacementResult) => void;
}

const ImageCard: React.FC<ImageCardProps> = ({
  image,
  processedImage,
  isProcessing,
  onReset,
  placement,
  onAdjust,
}) => {
  const [adjustMode, setAdjustMode] = useState(false);
  const [localPlacement, setLocalPlacement] = useState<PlacementResult | null>(placement || null);
  const adjustTimeoutRef = useRef<NodeJS.Timeout | null>(null);

  useEffect(() => {
    if (placement && !adjustMode) setLocalPlacement(placement);
  }, [placement, adjustMode]);

  const handleSliderChange = (key: keyof PlacementResult, value: number) => {
    if (!localPlacement) return;
    const next = { ...localPlacement, [key]: value };
    setLocalPlacement(next);

    // デバウンスして再合成処理を呼ぶ（スライダー操作中のカクつき防止）
    if (adjustTimeoutRef.current) clearTimeout(adjustTimeoutRef.current);
    adjustTimeoutRef.current = setTimeout(() => {
      if (onAdjust) onAdjust(next);
    }, 150);
  };

  return (
    <motion.section
      key="preview"
      initial={{ opacity: 0, scale: 0.98 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.4 }}
      style={{ width: '100%', display: 'flex', flexDirection: 'column', alignItems: 'center' }}
    >
      <div className="listing-card" id="image-card">
        <div className="listing-image-wrapper">
          <img
            src={processedImage || image}
            alt="処理対象の画像"
            className="listing-image"
          />

          {isProcessing && (
            <div className="processing-scrim">
              <motion.div
                animate={{ rotate: 360 }}
                transition={{ duration: 2, repeat: Infinity, ease: 'linear' }}
                className="processing-spinner"
              >
                <Sparkles size={40} />
              </motion.div>
              <div style={{ fontWeight: 600, fontSize: '16px' }}>
                猫を忍び込ませ中...
              </div>
            </div>
          )}
        </div>

        <div className="card-content">
          {adjustMode && localPlacement ? (
            <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="adjust-panel">
              <h3 style={{ fontSize: '14px', marginBottom: '14px', color: 'var(--text-primary)' }}>位置とサイズの微調整</h3>
              <div className="slider-row">
                <label>左右 (X)</label>
                <input type="range" min="0" max="1" step="0.01" value={localPlacement.x} onChange={e => handleSliderChange('x', parseFloat(e.target.value))} />
              </div>
              <div className="slider-row">
                <label>上下 (Y)</label>
                <input type="range" min="0" max="1" step="0.01" value={localPlacement.y} onChange={e => handleSliderChange('y', parseFloat(e.target.value))} />
              </div>
              <div className="slider-row">
                <label>大きさ</label>
                <input type="range" min="0.05" max="0.5" step="0.01" value={localPlacement.scale} onChange={e => handleSliderChange('scale', parseFloat(e.target.value))} />
              </div>
              <button className="btn-secondary" style={{ width: '100%', marginTop: '16px', justifyContent: 'center' }} onClick={() => setAdjustMode(false)}>
                <Check size={16} /> 確定する
              </button>
            </motion.div>
          ) : (
            <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
              <h2 className="card-title">カフェの思い出</h2>
              <p className="card-desc">しれっと忍び込んだ猫と一緒に。</p>

              <div className="actions-row">
                <button onClick={onReset} className="btn-circle" title="やり直す" id="btn-reset">
                  <RotateCcw size={20} />
                </button>
                {placement && (
                  <button onClick={() => setAdjustMode(true)} className="btn-circle" title="猫の調整" id="btn-adjust">
                    <SlidersHorizontal size={20} />
                  </button>
                )}
                <button className="btn-primary-red" style={{ flex: 1, justifyContent: 'center' }} id="btn-share">
                  <Share2 size={20} />
                  シェアする
                </button>
              </div>
            </motion.div>
          )}
        </div>
      </div>
    </motion.section>
  );
};

export default ImageCard;
