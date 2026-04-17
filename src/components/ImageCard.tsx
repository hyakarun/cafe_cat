import React from 'react';
import { Share2, RotateCcw, Sparkles, Cat } from 'lucide-react';
import { motion } from 'framer-motion';

interface ImageCardProps {
  image: string;
  processedImage: string | null;
  isProcessing: boolean;
  onReset: () => void;
}

const ImageCard: React.FC<ImageCardProps> = ({
  image,
  processedImage,
  isProcessing,
  onReset,
}) => (
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
        <h2 className="card-title">カフェの思い出</h2>
        <p className="card-desc">しれっと忍び込んだ猫と一緒に。</p>

        <div className="actions-row">
          <button onClick={onReset} className="btn-circle" title="やり直す" id="btn-reset">
            <RotateCcw size={20} />
          </button>
          <button className="btn-primary-red" style={{ flex: 1, justifyContent: 'center' }} id="btn-share">
            <Share2 size={20} />
            シェアする
          </button>
        </div>
      </div>
    </div>
  </motion.section>
);

export default ImageCard;
