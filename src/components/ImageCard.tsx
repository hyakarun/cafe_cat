import React, { useState, useEffect, useRef } from 'react';
import { Share2, RotateCcw, Sparkles, SlidersHorizontal, Check } from 'lucide-react';
import { motion } from 'framer-motion';
import type { PlacementResult } from '../lib/pipeline';

interface ImageCardProps {
  image: string;
  processedImage: string | null;
  debugImage?: string; // AI認識用のデバッグ画像
  isProcessing: boolean;
  onReset: () => void;
  placement?: PlacementResult | null;
  onAdjust?: (newPlacement: PlacementResult, forceRerollAsset?: boolean) => void;
}

const ImageCard: React.FC<ImageCardProps> = ({
  image,
  processedImage,
  debugImage,
  isProcessing,
  onReset,
  placement,
  onAdjust,
}) => {
  const [adjustMode, setAdjustMode] = useState(false);
  const [showDebug, setShowDebug] = useState(false);
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

  const handleModeChange = (mode: 'side' | 'diagonal') => {
    if (!localPlacement) return;
    const next = { ...localPlacement, angleMode: mode };
    setLocalPlacement(next);
    if (onAdjust) onAdjust(next, true); // モード変更時は再抽選つきで即反映
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
            src={showDebug && debugImage ? debugImage : (processedImage || image)}
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
              <h3 style={{ fontSize: '14px', marginBottom: '14px', color: 'var(--text-primary)' }}>位置・アングル微調整</h3>

              <div style={{ display: 'flex', gap: '8px', marginBottom: '16px' }}>
                <button 
                  className="btn-secondary" 
                  style={{ flex: 1, padding: '8px', fontSize: '13px', border: localPlacement.angleMode === 'side' ? '2px solid var(--primary-color)' : '' }}
                  onClick={() => handleModeChange('side')}
                >📷 真横</button>
                <button 
                  className="btn-secondary" 
                  style={{ flex: 1, padding: '8px', fontSize: '13px', border: localPlacement.angleMode === 'diagonal' ? '2px solid var(--primary-color)' : '' }}
                  onClick={() => handleModeChange('diagonal')}
                >📷 斜め/俯瞰</button>
              </div>
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

              <div style={{ display: 'flex', gap: '8px', marginTop: '16px' }}>
                <button className="btn-secondary" style={{ flex: 1, justifyContent: 'center' }} onClick={() => { if(onAdjust) onAdjust(localPlacement, true); }}>
                  <Sparkles size={16} /> 別の猫にする
                </button>
                <button className="btn-primary-red" style={{ flex: 1, justifyContent: 'center' }} onClick={() => setAdjustMode(false)}>
                  <Check size={16} /> 完了
                </button>
              </div>
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
                {debugImage && (
                  <button onClick={() => setShowDebug(!showDebug)} className={`btn-circle ${showDebug ? 'active-debug' : ''}`} title="AIの認識領域を見る" style={{ background: showDebug ? 'var(--bg-secondary)' : '' }}>
                    <Sparkles size={20} />
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
