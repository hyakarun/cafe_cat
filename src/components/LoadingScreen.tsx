import React, { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

const STAGE_MESSAGES: Record<1 | 2 | 3, string[]> = {
  1: [
    'ノラネコを探しにいっています...',
    'ネコのしつけ中です...',
  ],
  2: [
    'ネコのご機嫌をとっています...',
    'ネコが脱走したので追いかけてます...',
  ],
  3: [
    '猫を配置中です...',
    '猫にじっとしてもらうようお願いしてます...',
  ],
};

function pickRandom(arr: string[]) {
  return arr[Math.floor(Math.random() * arr.length)];
}

const LoadingScreen: React.FC = () => {
  const [progress, setProgress] = useState(0);   // 0–100
  const [stage, setStage]       = useState<1 | 2 | 3>(1);
  const [message, setMessage]   = useState(pickRandom(STAGE_MESSAGES[1]));
  const prevStageRef = useRef<1 | 2 | 3>(1);

  /* プログレスバーをゆっくり進める */
  useEffect(() => {
    const id = setInterval(() => {
      setProgress(p => {
        // 95% で止まり、AI完了時に親が画面遷移する
        if (p >= 95) { clearInterval(id); return 95; }
        // 後半ほど遅くなるイーズ
        const step = p < 40 ? 1.8 : p < 75 ? 0.9 : 0.4;
        return Math.min(95, p + step);
      });
    }, 120);
    return () => clearInterval(id);
  }, []);

  /* 進行度に応じてステージ＆メッセージを更新 */
  useEffect(() => {
    const newStage: 1 | 2 | 3 = progress < 40 ? 1 : progress < 75 ? 2 : 3;
    if (newStage !== prevStageRef.current) {
      prevStageRef.current = newStage;
      setStage(newStage);
      setMessage(pickRandom(STAGE_MESSAGES[newStage]));
    }
  }, [progress]);

  /* 同じステージ内でも一定間隔でメッセージをローテーション */
  useEffect(() => {
    const id = setInterval(() => {
      setMessage(pickRandom(STAGE_MESSAGES[prevStageRef.current]));
    }, 3200);
    return () => clearInterval(id);
  }, []);

  const stageLabels: Record<1 | 2 | 3, string> = {
    1: '準備中',
    2: '解析中',
    3: '合成中',
  };

  return (
    <div className="loading-screen">
      {/* 肉球アニメーション */}
      <motion.div
        className="loading-paw"
        animate={{ rotate: [0, -18, 18, -10, 10, 0] }}
        transition={{ duration: 1.8, repeat: Infinity, ease: 'easeInOut' }}
      >
        🐾
      </motion.div>

      <p className="loading-title">猫を忍び込ませ中</p>

      {/* 3ステップ インジケーター */}
      <div className="loading-steps">
        {([1, 2, 3] as const).map(s => (
          <div key={s} className={`loading-step ${s < stage ? 'done' : s === stage ? 'active' : ''}`}>
            <div className="loading-step-dot" />
            <span className="loading-step-label">{stageLabels[s]}</span>
          </div>
        ))}
      </div>

      {/* プログレスバー */}
      <div className="loading-bar-bg">
        <motion.div
          className="loading-bar-fill"
          animate={{ width: `${progress}%` }}
          transition={{ ease: 'easeOut', duration: 0.3 }}
        />
      </div>

      {/* 動的メッセージ */}
      <AnimatePresence mode="wait">
        <motion.p
          key={message}
          className="loading-sub"
          initial={{ opacity: 0, y: 6 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -6 }}
          transition={{ duration: 0.3 }}
        >
          {message}
        </motion.p>
      </AnimatePresence>
    </div>
  );
};

export default LoadingScreen;
