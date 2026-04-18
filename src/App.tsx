import React, { useState, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { pipelineInstance } from './lib/pipeline';
import type { PipelineResult, PlacementResult } from './lib/pipeline';
import Navbar from './components/Navbar';
import HeroSection from './components/HeroSection';
import ImageCard from './components/ImageCard';
import DebugPanel from './components/DebugPanel';
import LoadingScreen from './components/LoadingScreen';
import './index.css';

type Screen = 'landing' | 'preview' | 'loading' | 'result';

const pageVariants = {
  initial: { opacity: 0, y: 16 },
  animate: { opacity: 1, y: 0, transition: { duration: 0.35, ease: [0.4, 0, 0.2, 1] } },
  exit:    { opacity: 0, y: -12, transition: { duration: 0.2 } },
};

const App: React.FC = () => {
  const [screen, setScreen]               = useState<Screen>('landing');
  const [image, setImage]                 = useState<string | null>(null);
  const [processedImage, setProcessedImage] = useState<string | null>(null);
  const [result, setResult]               = useState<PipelineResult | null>(null);

  /* ── 画像選択（確認画面へ） ── */
  const handleSelectImage = useCallback(() => {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = 'image/*';
    input.onchange = (e: Event) => {
      const file = (e.target as HTMLInputElement).files?.[0];
      if (!file) return;
      const reader = new FileReader();
      reader.onload = (re) => {
        const dataUrl = re.target?.result as string;
        setImage(dataUrl);
        setProcessedImage(null);
        setResult(null);
        setScreen('preview');   // ← 確認画面へ
      };
      reader.readAsDataURL(file);
    };
    input.click();
  }, []);

  /* ── AI処理を開始 ── */
  const handleProcess = useCallback(async () => {
    if (!image) return;
    setScreen('loading');
    try {
      const pipelineResult = await pipelineInstance.process(image);
      setProcessedImage(pipelineResult.imageDataUrl);
      setResult(pipelineResult);
      setScreen('result');
    } catch (error) {
      console.error('Processing failed:', error);
      setProcessedImage(image);
      setScreen('result');
    }
  }, [image]);

  /* ── リセット ── */
  const handleReset = useCallback(() => {
    setImage(null);
    setProcessedImage(null);
    setResult(null);
    setScreen('landing');
  }, []);

  /* ── 位置調整 ── */
  const handleAdjust = useCallback(async (newPlacement: PlacementResult) => {
    const newImage = await pipelineInstance.recompose(newPlacement);
    if (newImage) {
      setProcessedImage(newImage);
      setResult(prev => prev ? { ...prev, placement: newPlacement, imageDataUrl: newImage } : null);
    }
  }, []);

  return (
    <div className="airbnb-container">
      <Navbar />
      <main className="main-content">
        <AnimatePresence mode="wait">

          {/* ① ランディング */}
          {screen === 'landing' && (
            <motion.div key="landing" variants={pageVariants} initial="initial" animate="animate" exit="exit">
              <HeroSection onSelectImage={handleSelectImage} />
            </motion.div>
          )}

          {/* ② 写真確認画面 */}
          {screen === 'preview' && image && (
            <motion.div key="preview" variants={pageVariants} initial="initial" animate="animate" exit="exit"
              className="preview-screen">
              <p className="screen-label">この写真でよろしいですか？</p>
              <div className="preview-image-wrapper">
                <img src={image} alt="選択した写真" className="preview-image" />
              </div>
              <div className="preview-actions">
                <button className="btn-secondary" onClick={handleSelectImage}>
                  選び直す
                </button>
                <button className="btn-primary-red" onClick={handleProcess}>
                  猫を忍び込ませる
                </button>
              </div>
            </motion.div>
          )}

          {/* ③ ローディング */}
          {screen === 'loading' && (
            <motion.div key="loading" variants={pageVariants} initial="initial" animate="animate" exit="exit"
              style={{ width: '100%' }}>
              <LoadingScreen />
            </motion.div>
          )}

          {/* ④ 結果 */}
          {screen === 'result' && (
            <motion.div key="result" variants={pageVariants} initial="initial" animate="animate" exit="exit"
              style={{ width: '100%', display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
              <ImageCard
                image={image!}
                processedImage={processedImage}
                isProcessing={false}
                onReset={handleReset}
                placement={result?.placement}
                onAdjust={handleAdjust}
              />
              {result && (
                <DebugPanel
                  detectedLabels={result.detectedLabels}
                  placement={result.placement}
                  modelLoaded={result.modelLoaded}
                />
              )}
            </motion.div>
          )}

        </AnimatePresence>
      </main>

      <footer className="footer">Built with Edge AI &amp; WebGPU</footer>
    </div>
  );
};

export default App;
