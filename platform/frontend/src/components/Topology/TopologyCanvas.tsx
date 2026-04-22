import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import type { SitePosition, UEPosition } from '@/api/types';

interface Bounds {
  min_x: number;
  max_x: number;
  min_y: number;
  max_y: number;
}

interface Props {
  sites: SitePosition[];
  ues: UEPosition[];
  bounds: Bounds;
  cellRadius: number;
  readOnly?: boolean;
  onSiteDrag?: (sites: SitePosition[]) => void;
  onUEDrag?: (ues: UEPosition[]) => void;
}

const SITE_COLORS = [
  '#1677ff', '#52c41a', '#fa541c', '#722ed1',
  '#13c2c2', '#eb2f96', '#faad14', '#2f54eb',
  '#a0d911', '#f5222d', '#1890ff', '#597ef7',
];
const UE_COLOR = '#595959';
const UE_DRAG_COLOR = '#1677ff';
const HEX_FILL = 'rgba(22,119,255,0.04)';
const HEX_STROKE = 'rgba(22,119,255,0.25)';
const SECTOR_LINE = 'rgba(0,0,0,0.15)';
const PADDING = 40;

function hexPoints(cx: number, cy: number, r: number): string {
  const pts: string[] = [];
  for (let i = 0; i < 6; i++) {
    const angle = (Math.PI / 3) * i - Math.PI / 6;
    pts.push(`${cx + r * Math.cos(angle)},${cy + r * Math.sin(angle)}`);
  }
  return pts.join(' ');
}

function siteColor(siteId: number): string {
  return SITE_COLORS[siteId % SITE_COLORS.length];
}

export default function TopologyCanvas({
  sites,
  ues,
  bounds,
  cellRadius,
  readOnly = false,
  onSiteDrag,
  onUEDrag,
}: Props) {
  const svgRef = useRef<SVGSVGElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [size, setSize] = useState({ w: 600, h: 500 });
  const [drag, setDrag] = useState<{
    type: 'site' | 'ue';
    id: number;
    startX: number;
    startY: number;
  } | null>(null);
  const [localSites, setLocalSites] = useState<SitePosition[]>([]);
  const [localUEs, setLocalUEs] = useState<UEPosition[]>([]);
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [panning, setPanning] = useState(false);
  const panStart = useRef({ x: 0, y: 0, panX: 0, panY: 0 });

  useEffect(() => {
    setLocalSites(sites);
  }, [sites]);

  useEffect(() => {
    setLocalUEs(ues);
  }, [ues]);

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const obs = new ResizeObserver((entries) => {
      const { width, height } = entries[0].contentRect;
      setSize({ w: Math.max(width, 200), h: Math.max(height, 200) });
    });
    obs.observe(el);
    return () => obs.disconnect();
  }, []);

  const transform = useMemo(() => {
    const dataW = bounds.max_x - bounds.min_x || 1;
    const dataH = bounds.max_y - bounds.min_y || 1;
    const scaleX = (size.w - PADDING * 2) / dataW;
    const scaleY = (size.h - PADDING * 2) / dataH;
    const scale = Math.min(scaleX, scaleY) * zoom;
    const cx = (bounds.min_x + bounds.max_x) / 2;
    const cy = (bounds.min_y + bounds.max_y) / 2;
    return {
      toSvg: (x: number, y: number): [number, number] => [
        size.w / 2 + (x - cx) * scale + pan.x,
        size.h / 2 - (y - cy) * scale + pan.y,
      ],
      toData: (sx: number, sy: number): [number, number] => [
        (sx - size.w / 2 - pan.x) / scale + cx,
        -((sy - size.h / 2 - pan.y) / scale) + cy,
      ],
      scale,
    };
  }, [bounds, size, zoom, pan]);

  const hexR = cellRadius * transform.scale;

  const uniqueSiteCenters = useMemo(() => {
    const map = new Map<string, { x: number; y: number; siteId: number }>();
    for (const s of localSites) {
      const key = `${s.x.toFixed(1)}_${s.y.toFixed(1)}`;
      if (!map.has(key)) map.set(key, { x: s.x, y: s.y, siteId: s.site_id });
    }
    return Array.from(map.values());
  }, [localSites]);

  const handleMouseDown = useCallback(
    (type: 'site' | 'ue', id: number, e: React.MouseEvent) => {
      if (readOnly) return;
      e.stopPropagation();
      e.preventDefault();
      setDrag({ type, id, startX: e.clientX, startY: e.clientY });
    },
    [readOnly],
  );

  const handleSvgMouseDown = useCallback(
    (e: React.MouseEvent) => {
      if (drag) return;
      if ((e.button === 0 && (e.ctrlKey || e.metaKey)) || e.button === 1) {
        setPanning(true);
        panStart.current = { x: e.clientX, y: e.clientY, panX: pan.x, panY: pan.y };
      }
    },
    [drag, pan],
  );

  const handleMouseMove = useCallback(
    (e: React.MouseEvent) => {
      if (panning) {
        const dx = e.clientX - panStart.current.x;
        const dy = e.clientY - panStart.current.y;
        setPan({ x: panStart.current.panX + dx, y: panStart.current.panY + dy });
        return;
      }
      if (!drag) return;
      const svg = svgRef.current;
      if (!svg) return;
      const rect = svg.getBoundingClientRect();
      const sx = e.clientX - rect.left;
      const sy = e.clientY - rect.top;
      const [dx, dy] = transform.toData(sx, sy);

      if (drag.type === 'site') {
        setLocalSites((prev) => {
          const origSite = sites.find(
            (s) => s.site_id === drag.id && s.sector_id === 0,
          );
          if (!origSite) return prev;
          const offsetX = dx - origSite.x;
          const offsetY = dy - origSite.y;
          return prev.map((s) =>
            s.site_id === drag.id
              ? { ...s, x: sites.find((os) => os.site_id === s.site_id && os.sector_id === s.sector_id)!.x + offsetX, y: sites.find((os) => os.site_id === s.site_id && os.sector_id === s.sector_id)!.y + offsetY }
              : s,
          );
        });
      } else {
        setLocalUEs((prev) =>
          prev.map((u) => (u.ue_id === drag.id ? { ...u, x: dx, y: dy } : u)),
        );
      }
    },
    [drag, panning, transform, sites],
  );

  const handleMouseUp = useCallback(() => {
    if (panning) {
      setPanning(false);
      return;
    }
    if (!drag) return;
    if (drag.type === 'site') onSiteDrag?.(localSites);
    else onUEDrag?.(localUEs);
    setDrag(null);
  }, [drag, panning, localSites, localUEs, onSiteDrag, onUEDrag]);

  const handleWheel = useCallback((e: React.WheelEvent) => {
    e.preventDefault();
    setZoom((z) => Math.max(0.2, Math.min(5, z - e.deltaY * 0.001)));
  }, []);

  return (
    <div
      ref={containerRef}
      style={{ width: '100%', height: '100%', minHeight: 300, position: 'relative', overflow: 'hidden' }}
    >
      <svg
        ref={svgRef}
        width={size.w}
        height={size.h}
        style={{ position: 'absolute', top: 0, left: 0, cursor: drag ? 'grabbing' : panning ? 'grabbing' : 'default', userSelect: 'none' }}
        onMouseDown={handleSvgMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
        onWheel={handleWheel}
      >
        {/* Grid background */}
        <defs>
          <pattern id="grid" width="40" height="40" patternUnits="userSpaceOnUse">
            <path d="M 40 0 L 0 0 0 40" fill="none" stroke="#f0f0f0" strokeWidth="0.5" />
          </pattern>
        </defs>
        <rect width={size.w} height={size.h} fill="url(#grid)" />

        {/* Hex cells */}
        {uniqueSiteCenters.map((c) => {
          const [sx, sy] = transform.toSvg(c.x, c.y);
          return (
            <polygon
              key={`hex-${c.siteId}`}
              points={hexPoints(sx, sy, hexR)}
              fill={HEX_FILL}
              stroke={HEX_STROKE}
              strokeWidth={1}
            />
          );
        })}

        {/* Sector lines */}
        {localSites.map((s) => {
          const [sx, sy] = transform.toSvg(s.x, s.y);
          const angle = ((90 - s.azimuth_deg) * Math.PI) / 180;
          const len = hexR * 0.85;
          const ex = sx + len * Math.cos(angle);
          const ey = sy - len * Math.sin(angle);
          return (
            <line
              key={`sec-${s.site_id}-${s.sector_id}`}
              x1={sx}
              y1={sy}
              x2={ex}
              y2={ey}
              stroke={SECTOR_LINE}
              strokeWidth={1}
              strokeDasharray="4 2"
            />
          );
        })}

        {/* UE dots */}
        {localUEs.map((u) => {
          const [sx, sy] = transform.toSvg(u.x, u.y);
          const isDragging = drag?.type === 'ue' && drag.id === u.ue_id;
          return (
            <g key={`ue-${u.ue_id}`}>
              <circle
                cx={sx}
                cy={sy}
                r={isDragging ? 5 : 3}
                fill={isDragging ? UE_DRAG_COLOR : UE_COLOR}
                opacity={isDragging ? 1 : 0.5}
                style={{ cursor: readOnly ? 'default' : 'grab' }}
                onMouseDown={(e) => handleMouseDown('ue', u.ue_id, e)}
              />
            </g>
          );
        })}

        {/* Site towers */}
        {uniqueSiteCenters.map((c) => {
          const [sx, sy] = transform.toSvg(c.x, c.y);
          const color = siteColor(c.siteId);
          const isDragging = drag?.type === 'site' && drag.id === c.siteId;
          return (
            <g
              key={`site-${c.siteId}`}
              style={{ cursor: readOnly ? 'default' : 'grab' }}
              onMouseDown={(e) => handleMouseDown('site', c.siteId, e)}
            >
              {/* Outer ring */}
              <circle
                cx={sx}
                cy={sy}
                r={isDragging ? 12 : 10}
                fill="white"
                stroke={color}
                strokeWidth={isDragging ? 3 : 2}
              />
              {/* Tower icon */}
              <circle cx={sx} cy={sy} r={4} fill={color} />
              {/* Label */}
              <text
                x={sx}
                y={sy - 15}
                textAnchor="middle"
                fontSize={10}
                fontWeight={600}
                fill={color}
              >
                BS{c.siteId}
              </text>
            </g>
          );
        })}

        {/* Legend */}
        <g transform={`translate(${size.w - 130}, 12)`}>
          <rect x={0} y={0} width={120} height={readOnly ? 52 : 76} rx={4} fill="rgba(255,255,255,0.9)" stroke="#e8e8e8" />
          <circle cx={14} cy={16} r={5} fill="#1677ff" stroke="white" strokeWidth={1.5} />
          <text x={26} y={20} fontSize={11} fill="#333">基站</text>
          <circle cx={14} cy={36} r={3} fill={UE_COLOR} opacity={0.6} />
          <text x={26} y={40} fontSize={11} fill="#333">终端 (UE)</text>
          {!readOnly && (
            <text x={8} y={60} fontSize={9} fill="#999">
              拖拽可调整位置
            </text>
          )}
        </g>

        {/* Zoom controls */}
        {!readOnly && (
          <g transform={`translate(12, ${size.h - 80})`}>
            <rect
              x={0}
              y={0}
              width={28}
              height={60}
              rx={4}
              fill="rgba(255,255,255,0.9)"
              stroke="#e8e8e8"
            />
            <text
              x={14}
              y={20}
              textAnchor="middle"
              fontSize={16}
              fill="#333"
              style={{ cursor: 'pointer', userSelect: 'none' }}
              onClick={() => setZoom((z) => Math.min(5, z + 0.2))}
            >
              +
            </text>
            <line x1={4} y1={30} x2={24} y2={30} stroke="#e8e8e8" />
            <text
              x={14}
              y={48}
              textAnchor="middle"
              fontSize={16}
              fill="#333"
              style={{ cursor: 'pointer', userSelect: 'none' }}
              onClick={() => setZoom((z) => Math.max(0.2, z - 0.2))}
            >
              −
            </text>
          </g>
        )}

        {/* Scale bar */}
        {transform.scale > 0 && (
          <g transform={`translate(${readOnly ? 12 : 52}, ${size.h - 24})`}>
            {(() => {
              const targetPx = 80;
              const mPerPx = 1 / transform.scale;
              const rawM = targetPx * mPerPx;
              const niceM =
                rawM >= 1000
                  ? Math.round(rawM / 500) * 500
                  : rawM >= 100
                    ? Math.round(rawM / 100) * 100
                    : rawM >= 10
                      ? Math.round(rawM / 10) * 10
                      : Math.round(rawM);
              const barPx = niceM > 0 ? niceM * transform.scale : 60;
              const label = niceM >= 1000 ? `${niceM / 1000} km` : `${niceM} m`;
              return (
                <>
                  <line x1={0} y1={0} x2={barPx} y2={0} stroke="#666" strokeWidth={2} />
                  <line x1={0} y1={-4} x2={0} y2={4} stroke="#666" strokeWidth={1.5} />
                  <line x1={barPx} y1={-4} x2={barPx} y2={4} stroke="#666" strokeWidth={1.5} />
                  <text x={barPx / 2} y={-6} textAnchor="middle" fontSize={10} fill="#666">
                    {label}
                  </text>
                </>
              );
            })()}
          </g>
        )}
      </svg>
    </div>
  );
}
