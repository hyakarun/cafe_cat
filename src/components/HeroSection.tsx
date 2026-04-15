import React from 'react';
import { Camera, Upload } from 'lucide-react';
import { motion } from 'framer-motion';

interface HeroSectionProps {
  onSelectImage: () => void;
}

const HeroSection: React.FC<HeroSectionProps> = ({ onSelectImage }) => (
  <motion.section
    key="landing"
    className="hero-section"
    initial={{ opacity: 0, y: 10 }}
    animate={{ opacity: 1, y: 0 }}
    exit={{ opacity: 0, y: -10 }}
    transition={{ duration: 0.3 }}
  >
    <h1 className="headline">
      日常に、<br />しれっと猫を。
    </h1>
    <p className="subheadline">
      AIがあなたのカフェ写真に、ミニマルな猫をこっそり忍び込ませます。
    </p>

    <div className="buttons-group">
      <button onClick={onSelectImage} className="btn-primary-red" id="btn-capture">
        <Camera size={20} />
        写真を撮る
      </button>
      <button onClick={onSelectImage} className="btn-secondary" id="btn-upload">
        <Upload size={20} />
        ライブラリから選択
      </button>
    </div>
  </motion.section>
);

export default HeroSection;
