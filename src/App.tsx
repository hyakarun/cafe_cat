import React, { useState } from 'react';
import { Camera, Upload, Share2, Sparkles, Cat, Info, ChevronRight } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { pipelineInstance } from './lib/pipeline';
import './index.css';

const App: React.FC = () => {
  const [image, setImage] = useState<string | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [processedImage, setProcessedImage] = useState<string | null>(null);
  const [debugInfo, setDebugInfo] = useState<any>(null);
  
  const handleCapture = () => {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = 'image/*';
    input.onchange = (e: any) => {
      const file = e.target.files?.[0];
      if (file) {
        const reader = new FileReader();
        reader.onload = async (re) => {
          const res = re.target?.result as string;
          setImage(res);
          await processImage(res);
        };
        reader.readAsDataURL(file);
      }
    };
    input.click();
  };

  const processImage = async (img: string) => {
    setIsProcessing(true);
    setDebugInfo(null);
    try {
      const { result, debugInfo } = await pipelineInstance.process(img);
      setProcessedImage(result);
      setDebugInfo(debugInfo);
    } catch (error) {
      console.error('Processing failed:', error);
      setProcessedImage(img);
    } finally {
      setIsProcessing(false);
    }
  };

  const reset = () => {
    setImage(null);
    setProcessedImage(null);
    setIsProcessing(false);
  };

  return (
    <div className="app-container">
      <header className="nav-glass">
        <div className="nav-logo">
          <Cat size={18} />
          <span>Shiretto Cat</span>
        </div>
        <button className="nav-icon">
          <Info size={18} />
        </button>
      </header>

      <main className="main-content">
        <AnimatePresence mode="wait">
          {!image ? (
            <motion.section
              key="landing"
              className="section-dark"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
            >
              <h1 className="hero-headline">日常に、<br/>しれっと猫を。</h1>
              <p className="hero-subheadline">
                AIがあなたのカフェ写真に、ミニマルな猫をこっそり忍び込ませます。
              </p>

              <div className="buttons-row mt-6">
                <button onClick={handleCapture} className="btn-primary">
                  <Camera size={18} />
                  写真を撮る
                </button>
                <button onClick={handleCapture} className="btn-secondary-link">
                  <Upload size={18} />
                  ライブラリから選択
                </button>
              </div>
            </motion.section>
          ) : (
            <motion.section
              key="preview"
              className="section-light"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
            >
              <div className="card-container">
                <img src={image} alt="Target" className="preview-image" />
                
                {isProcessing && (
                  <div className="processing-overlay">
                    <motion.div animate={{ rotate: 360 }} transition={{ duration: 2, repeat: Infinity, ease: "linear" }}>
                      <Sparkles size={32} />
                    </motion.div>
                    <p className="processing-text">猫を忍び込ませ中...</p>
                    <div className="progress-bar-container">
                      <motion.div 
                        className="progress-bar"
                        initial={{ width: 0 }}
                        animate={{ width: "100%" }}
                        transition={{ duration: 2 }}
                      />
                    </div>
                  </div>
                )}

                {processedImage && !isProcessing && (
                  <motion.div 
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    className="processed-icon-overlay"
                  >
                    <Cat size={64} color="#ffffff" />
                  </motion.div>
                )}
              </div>

              <div className="buttons-row mt-6">
                <button onClick={reset} className="btn-link">
                  やり直す <ChevronRight size={16} />
                </button>
                <button className="btn-primary">
                  <Share2 size={18} />
                  シェアする
                </button>
              </div>

              {debugInfo && (
                <div className="debug-container">
                  <div style={{ color: 'rgba(255,255,255,0.48)', marginBottom: '8px', textTransform: 'uppercase' }}>Debug Information</div>
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px' }}>
                    <div>
                      <div style={{ color: 'rgba(255,255,255,0.8)' }}>Objects:</div>
                      <div>{debugInfo.labels.join(', ')}</div>
                    </div>
                    <div>
                      <div style={{ color: 'var(--apple-blue)' }}>Placement: {debugInfo.placement.reason}</div>
                      <div style={{ fontSize: '10px', color: 'rgba(255,255,255,0.48)' }}>
                        X: {debugInfo.placement.x.toFixed(2)}, Y: {debugInfo.placement.y.toFixed(2)}
                      </div>
                    </div>
                  </div>
                </div>
              )}
            </motion.section>
          )}
        </AnimatePresence>
      </main>

      <footer className="footer">
        Built with Edge AI & WebGPU
      </footer>
    </div>
  );
};

export default App;
