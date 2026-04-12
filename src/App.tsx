import React, { useState } from 'react';
import { Camera, Upload, Share2, Sparkles, Cat, Info } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { pipelineInstance } from './lib/pipeline';

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
    <div className="min-h-screen flex flex-col p-6">
      <header className="flex justify-between items-center mb-10">
        <motion.div 
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          className="flex items-center gap-2"
        >
          <Cat size={32} className="text-white" />
          <h1 className="text-xl font-bold tracking-tighter">Shiretto Cat</h1>
        </motion.div>
        <motion.button
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          className="p-2 rounded-full glass"
        >
          <Info size={20} />
        </motion.button>
      </header>

      <main className="flex-1 flex flex-col items-center justify-center">
        <AnimatePresence mode="wait">
          {!image ? (
            <motion.div
              key="landing"
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.9 }}
              className="text-center"
            >
              <div className="w-64 h-64 mb-10 mx-auto glass flex items-center justify-center relative overflow-hidden">
                <motion.div
                  animate={{ 
                    y: [0, -10, 0],
                    rotate: [0, 2, 0]
                  }}
                  transition={{ duration: 4, repeat: Infinity, ease: "easeInOut" }}
                >
                  <Sparkles size={80} className="text-white/20" />
                </motion.div>
                <div className="absolute inset-0 bg-gradient-to-t from-white/5 to-transparent" />
              </div>
              
              <h2 className="text-2xl font-bold mb-4">日常に、しれっと猫を。</h2>
              <p className="text-white/60 mb-10 max-w-xs mx-auto">
                AIがあなたのカフェ写真に、ミニマルな猫をこっそり忍び込ませます。
              </p>

              <button 
                onClick={handleCapture}
                className="primary-button flex items-center gap-2"
              >
                <Camera size={20} />
                写真を撮る
              </button>
              
              <button 
                onClick={handleCapture}
                className="mt-6 text-white/40 hover:text-white/80 transition-colors flex items-center gap-2 mx-auto"
              >
                <Upload size={16} />
                ライブラリから選択
              </button>
            </motion.div>
          ) : (
            <motion.div
              key="preview"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="w-full flex flex-col gap-6"
            >
              <div className="aspect-[3/4] rounded-3xl overflow-hidden glass relative">
                <img src={image} alt="Target" className="w-full h-full object-cover" />
                
                {isProcessing && (
                  <div className="absolute inset-0 bg-black/60 backdrop-blur-sm flex flex-col items-center justify-center">
                    <motion.div
                      animate={{ rotate: 360 }}
                      transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
                      className="mb-4"
                    >
                      <Sparkles size={48} className="text-white" />
                    </motion.div>
                    <p className="font-bold tracking-widest text-sm">猫を忍び込ませ中...</p>
                    
                    <div className="w-48 h-1 bg-white/10 rounded-full mt-6 overflow-hidden">
                      <motion.div 
                        className="h-full bg-white"
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
                    className="absolute bottom-10 right-10"
                  >
                    {/* Minimalist Cat Icon as Overlay Example */}
                    <div className="relative group">
                      <Cat size={64} className="text-white filter drop-shadow-lg" />
                    </div>
                  </motion.div>
                )}
              </div>

              <div className="flex gap-4">
                <button 
                  onClick={reset}
                  className="flex-1 p-4 rounded-3xl glass font-bold hover:bg-white/10 transition-colors"
                >
                  やり直す
                </button>
                <button 
                  className="flex-1 p-4 rounded-3xl bg-white text-black font-bold flex items-center justify-center gap-2 hover:opacity-90 transition-opacity"
                >
                  <Share2 size={20} />
                  シェアする
                </button>
              </div>

              {debugInfo && (
                <motion.div 
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  className="p-4 rounded-2xl bg-white/5 border border-white/10 text-xs font-mono"
                >
                  <div className="text-white/40 mb-2 uppercase tracking-widest text-[10px]">Debug Information</div>
                  <div className="grid grid-cols-2 gap-2">
                    <div>
                      <p className="text-white/60 mb-1">Detected Objects:</p>
                      <div className="flex flex-wrap gap-1">
                        {debugInfo.labels.map((l: string, i: number) => (
                          <span key={i} className="px-1.5 py-0.5 rounded bg-white/10 text-white/80">{l}</span>
                        ))}
                      </div>
                    </div>
                    <div>
                      <p className="text-white/60 mb-1">Placement:</p>
                      <p className="text-blue-400">{debugInfo.placement.reason}</p>
                      <p className="text-white/30 text-[9px]">X: {debugInfo.placement.x.toFixed(2)}, Y: {debugInfo.placement.y.toFixed(2)}</p>
                    </div>
                  </div>
                </motion.div>
              )}
            </motion.div>
          )}
        </AnimatePresence>
      </main>

      <footer className="mt-10 py-6 text-center text-[10px] text-white/20 tracking-[0.2em] uppercase">
        Built with Edge AI & WebGPU
      </footer>
    </div>
  );
};

export default App;
