"use client";

export default function LogoPage() {
  return (
    <div className="min-h-screen flex flex-col items-center justify-center gap-8 p-4">
      <h1 className="text-2xl font-bold">Logo Preview</h1>

      {/* Logo Preview */}
      <div className="p-8 border rounded-lg">
        <Logo />
      </div>

      {/* Download Link */}
      <a
        href="#"
        onClick={(e) => {
          e.preventDefault();
          const svg = document.querySelector("svg");
          if (!svg) return;

          const svgData = new XMLSerializer().serializeToString(svg);
          const blob = new Blob([svgData], { type: "image/svg+xml" });
          const url = URL.createObjectURL(blob);

          const link = document.createElement("a");
          link.href = url;
          link.download = "loldraftai-logo.svg";
          document.body.appendChild(link);
          link.click();
          document.body.removeChild(link);
          URL.revokeObjectURL(url);
        }}
        className="text-primary hover:underline cursor-pointer"
      >
        Download SVG
      </a>
    </div>
  );
}

function Logo() {
  // Fixed size of 512px
  const width = 512;
  const height = 512;
  const centerX = width / 2;
  const centerY = height / 2;
  const size = Math.min(width, height) * 0.48;

  // Calculate hexagon points
  const points = [];
  for (let i = 0; i < 6; i++) {
    const angle = ((2 * Math.PI) / 6) * i - Math.PI / 6; // Rotate 30 degrees to point up
    const x = centerX + size * Math.cos(angle);
    const y = centerY + size * Math.sin(angle);
    points.push(`${x},${y}`);
  }

  // Calculate 2 points for the lines - increased vertical spread to 0.4
  const leftPoints = [
    { x: centerX - size * 0.4, y: centerY - size * 0.4 },
    { x: centerX - size * 0.4, y: centerY + size * 0.4 },
  ];
  const rightX = centerX + size * 0.4;
  const rightY = centerY;

  return (
    <svg
      width={width}
      height={height}
      viewBox={`0 0 ${width} ${height}`}
      style={{ display: "block" }}
    >
      {/* Gradient definition */}
      <defs>
        <linearGradient id="brandGradient" x1="0%" y1="0%" x2="100%" y2="0%">
          <stop
            offset="0%"
            style={{ stopColor: "hsl(220, 70%, 50%)", stopOpacity: 1 }}
          />
          <stop
            offset="100%"
            style={{ stopColor: "hsl(160, 60%, 45%)", stopOpacity: 1 }}
          />
        </linearGradient>
      </defs>

      {/* Main hexagon shape */}
      <polygon
        points={points.join(" ")}
        fill="url(#brandGradient)"
        stroke="url(#brandGradient)"
        strokeWidth="8"
      />

      {/* Converging lines - increased strokeWidth to 32 */}
      {leftPoints.map((point, i) => (
        <line
          key={i}
          x1={point.x}
          y1={point.y}
          x2={rightX}
          y2={rightY}
          stroke="white"
          strokeWidth="32"
          strokeLinecap="round"
        />
      ))}

      {/* Starting points dots - increased radius to 32 */}
      {leftPoints.map((point, i) => (
        <circle key={i} cx={point.x} cy={point.y} r="32" fill="white" />
      ))}

      {/* End point dot - increased radius to 48 */}
      <circle cx={rightX} cy={rightY} r="48" fill="white" />
    </svg>
  );
}
