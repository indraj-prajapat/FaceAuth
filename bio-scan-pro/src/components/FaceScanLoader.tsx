import { Scan, Eye, Shield, Search, Activity } from "lucide-react";

export const FaceScanLoader = () => {
  return (
    <div className="fixed inset-0 bg-background/80 backdrop-blur-sm z-50 flex items-center justify-center">
      <div className="relative">
        {/* Outer scanning ring */}
        <div className="absolute inset-0 flex items-center justify-center">
          <div className="w-48 h-48 border-4 border-primary/30 rounded-full animate-ping" />
        </div>
        
        {/* Middle rotating ring */}
        <div className="absolute inset-0 flex items-center justify-center animate-spin-slow">
          <div className="w-40 h-40 border-t-4 border-r-4 border-primary rounded-full" />
        </div>

        {/* Inner face scan grid */}
        <div className="relative w-32 h-32 flex items-center justify-center">
          <div className="absolute inset-0 grid grid-cols-8 grid-rows-8 gap-1 opacity-50">
            {Array.from({ length: 64 }).map((_, i) => (
              <div
                key={i}
                className="bg-primary rounded-sm animate-pulse"
                style={{ animationDelay: `${i * 20}ms` }}
              />
            ))}
          </div>
          
          {/* Center icon */}
          <div className="relative z-10">
            <div className="absolute inset-0 bg-primary/20 blur-xl animate-pulse-glow" />
            <Scan className="relative h-10 w-10 text-primary animate-pulse" />
          </div>
        </div>

        {/* Status indicators around the circle */}
        <div className="absolute -top-16 left-1/2 -translate-x-1/2 flex items-center gap-2 animate-fade-in">
          <Eye className="h-3 w-3 text-primary" />
          <span className="text-xs text-muted-foreground">Face Detection</span>
        </div>
        
        <div className="absolute -right-24 top-1/2 -translate-y-1/2 flex items-center gap-2 animate-fade-in" style={{ animationDelay: '200ms' }}>
          <Shield className="h-3 w-3 text-primary" />
          <span className="text-xs text-muted-foreground">Quality Check</span>
        </div>
        
        <div className="absolute -bottom-16 left-1/2 -translate-x-1/2 flex items-center gap-2 animate-fade-in" style={{ animationDelay: '400ms' }}>
          <Search className="h-3 w-3 text-primary" />
          <span className="text-xs text-muted-foreground">DB Search</span>
        </div>
        
        <div className="absolute -left-24 top-1/2 -translate-y-1/2 flex items-center gap-2 animate-fade-in" style={{ animationDelay: '600ms' }}>
          <Activity className="h-3 w-3 text-primary" />
          <span className="text-xs text-muted-foreground">Risk Analysis</span>
        </div>
      </div>
    </div>
  );
};
