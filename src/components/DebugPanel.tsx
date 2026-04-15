import React from 'react';
import type { PlacementResult } from '../lib/pipeline';

interface DebugPanelProps {
  detectedLabels: string[];
  placement: PlacementResult;
  modelLoaded: boolean;
}

const DebugPanel: React.FC<DebugPanelProps> = ({
  detectedLabels,
  placement,
  modelLoaded,
}) => (
  <div className="debug-panel" id="debug-panel">
    <div style={{ fontWeight: 700, marginBottom: '8px', color: 'var(--text-primary)' }}>
      Debug Information
    </div>
    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px' }}>
      <div>
        <div style={{ color: 'var(--text-primary)' }}>検出オブジェクト:</div>
        <div>{detectedLabels.length > 0 ? detectedLabels.join(', ') : 'なし'}</div>
      </div>
      <div>
        <div style={{ color: 'var(--text-primary)' }}>配置:</div>
        <div>{placement.reason}</div>
        <div style={{ fontSize: '10px' }}>
          X: {placement.x.toFixed(2)}, Y: {placement.y.toFixed(2)}
        </div>
      </div>
      <div>
        <div style={{ color: 'var(--text-primary)' }}>モデル:</div>
        <div>{modelLoaded ? '✅ 読み込み済み' : '⚠️ フォールバック'}</div>
      </div>
      <div>
        <div style={{ color: 'var(--text-primary)' }}>スケール:</div>
        <div>{(placement.scale * 100).toFixed(0)}%</div>
      </div>
    </div>
  </div>
);

export default DebugPanel;
