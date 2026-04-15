import React, { useState, useCallback } from 'react';
import { AnimatePresence } from 'framer-motion';
import { pipelineInstance } from './lib/pipeline';
import type { PipelineResult } from './lib/pipeline';
import Navbar from './components/Navbar';
import HeroSection from './components/HeroSection';
import ImageCard from './components/ImageCard';
import DebugPanel from './components/DebugPanel';
import './index.css';

const App: React.FC = () => {
  const [image, setImage] = useState<string | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [processedImage, setProcessedImage] = useState<string | null>(null);
  const [result, setResult] = useState<PipelineResult | null>(null);

  /** 画像選択ダイアログを開く */
  const handleSelectImage = useCallback(() => {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = 'image/*';
    input.onchange = (e: Event) => {
      const file = (e.target as HTMLInputElement).files?.[0];
      if (!file) return;

      const reader = new FileReader();
      reader.onload = async (re) => {
        const dataUrl = re.target?.result as string;
        setImage(dataUrl);
        setProcessedImage(null);
        setResult(null);
        await processImage(dataUrl);
      };
      reader.readAsDataURL(file);
    };
    input.click();
  }, []);

  /** AI画像処理を実行 */
  const processImage = async (img: string) => {
    setIsProcessing(true);
    try {
      const pipelineResult = await pipelineInstance.process(img);
      setProcessedImage(pipelineResult.imageDataUrl);
      setResult(pipelineResult);
    } catch (error) {
      console.error('Processing failed:', error);
      setProcessedImage(img);
    } finally {
      setIsProcessing(false);
    }
  };

  /** リセット */
  const handleReset = useCallback(() => {
    setImage(null);
    setProcessedImage(null);
    setIsProcessing(false);
    setResult(null);
  }, []);

  return (
    <div className="airbnb-container">
      <Navbar />

      <main className="main-content">
        <AnimatePresence mode="wait">
          {!image ? (
            <HeroSection onSelectImage={handleSelectImage} />
          ) : (
            <>
              <ImageCard
                image={image}
                processedImage={processedImage}
                isProcessing={isProcessing}
                onReset={handleReset}
              />
              {result && !isProcessing && (
                <DebugPanel
                  detectedLabels={result.detectedLabels}
                  placement={result.placement}
                  modelLoaded={result.modelLoaded}
                />
              )}
            </>
          )}
        </AnimatePresence>
      </main>

      <footer className="footer">
        Built with Edge AI &amp; WebGPU
      </footer>
    </div>
  );
};

export default App;
