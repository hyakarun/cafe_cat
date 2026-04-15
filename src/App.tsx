import React, { useState } from 'react';
import { Camera, Upload, Share2, Sparkles, Cat, Info, RotateCcw } from 'lucide-react';
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
    <div className="airbnb-container">
      <header className="nav-bar">
        <div className="nav-brand">
          <Cat size={24} />
          <span>Shiretto Cat</span>
        </div>
        <button className="nav-circular-btn">
          <Info size={20} />
        </button>
      </header>

      <main className="main-content">
        <AnimatePresence mode="wait">
          {!image ? (
            <motion.section
              key="landing"
              className="hero-section"
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              transition={{ duration: 0.3 }}
            >
              <h1 className="headline">日常に、<br/>しれっと猫を。</h1>
              <p className="subheadline">
                AIがあなたのカフェ写真に、ミニマルな猫をこっそり忍び込ませます。
              </p>

              <div className="buttons-group">
                <button onClick={handleCapture} className="btn-primary-red">
                  <Camera size={20} />
                  写真を撮る
                </button>
                <button onClick={handleCapture} className="btn-secondary">
                  <Upload size={20} />
                  ライブラリから選択
                </button>
              </div>
            </motion.section>
          ) : (
            <motion.section
              key="preview"
              initial={{ opacity: 0, scale: 0.98 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ duration: 0.4 }}
              style={{ width: '100%', display: 'flex', flexDirection: 'column', alignItems: 'center' }}
            >
              <div className="listing-card">
                <div className="listing-image-wrapper">
                  <img src={processedImage || image} alt="Target" className="listing-image" />
                  {isProcessing && (
                    <div className="processing-scrim">
                      <motion.div 
                        animate={{ rotate: 360 }} 
                        transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
                        className="processing-spinner"
                      >
                        <Sparkles size={40} />
                      </motion.div>
                      <div style={{ fontWeight: 600, fontSize: '16px' }}>
                        猫を忍び込ませ中...
                      </div>
                    </div>
                  )}

                  {processedImage && !isProcessing && (
                    <motion.div 
                      initial={{ scale: 0, opacity: 0 }}
                      animate={{ scale: 1, opacity: 1 }}
                      transition={{ type: "spring", bounce: 0.5 }}
                      className="processed-element"
                    >
                      <Cat size={40} color="var(--brand-red)" />
                    </motion.div>
                  )}
                </div>

                <div className="card-content">
                  <h2 className="card-title">カフェの思い出</h2>
                  <p className="card-desc">しれっと忍び込んだ猫と一緒に。</p>
                  
                  <div className="actions-row">
                    <button onClick={reset} className="btn-circle" title="やり直す">
                      <RotateCcw size={20} />
                    </button>
                    <button className="btn-primary-red" style={{ flex: 1, justifyContent: 'center' }}>
                      <Share2 size={20} />
                      シェアする
                    </button>
                  </div>
                </div>
              </div>

              {debugInfo && (
                <div className="debug-panel">
                  <div style={{ fontWeight: 700, marginBottom: '8px', color: 'var(--text-primary)' }}>Debug Information</div>
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px' }}>
                    <div>
                      <div style={{ color: 'var(--text-primary)' }}>Targets Info:</div>
                      <div>Elements: {debugInfo.segOutputLength || 0}</div>
                      <div>Main Target: {debugInfo.targetLabel || 'None'}</div>
                    </div>
                    <div>
                      <div style={{ color: 'var(--text-primary)' }}>Placement:</div>
                      <div>{debugInfo.reason || 'No reason'}</div>
                      <div style={{ fontSize: '10px' }}>X: {debugInfo.x?.toFixed(2) ?? 0}, Y: {debugInfo.y?.toFixed(2) ?? 0}</div>
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
