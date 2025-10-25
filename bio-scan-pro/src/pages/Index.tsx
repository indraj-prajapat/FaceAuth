import { useState, useEffect } from "react";
import { Shield, Database, AlertCircle, CheckCircle, Clock, Scan, Upload, ClipboardList } from "lucide-react";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { StatsCard } from "@/components/StatsCard";
import { VerifySection } from "@/components/VerifySection";
import { BatchSection } from "@/components/BatchSection";
import { ReviewSection } from "@/components/ReviewSection";
import { useToast } from "@/hooks/use-toast";

const API_BASE = import.meta.env.VITE_API_BASE || "http://127.0.0.1:8000";

const Index = () => {
  const [stats, setStats] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [verifyOpen, setVerifyOpen] = useState(false);
  const [batchOpen, setBatchOpen] = useState(false);
  const [reviewOpen, setReviewOpen] = useState(false);
  const { toast } = useToast();

  const fetchStats = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/stats`);
      if (!res.ok) throw new Error("Failed to fetch stats");
      const data = await res.json();
      setStats(data);
    } catch (error) {
      toast({
        title: "Error",
        description: "Failed to load statistics",
        variant: "destructive",
      });
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchStats();
    const interval = setInterval(fetchStats, 30000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="h-screen w-screen bg-background overflow-hidden flex flex-col">
      {/* Hero Section */}
      <section className="relative bg-gradient-hero border-b border-border overflow-hidden flex-shrink-0">
        <div className="absolute inset-0 bg-[url('data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNjAiIGhlaWdodD0iNjAiIHZpZXdCb3g9IjAgMCA2MCA2MCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48ZyBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPjxwYXRoIGQ9Ik0zNiAxOGMzLjMxNCAwIDYgMi42ODYgNiA2cy0yLjY4NiA2LTYgNi02LTIuNjg2LTYtNiAyLjY4Ni02IDYtNiIgc3Ryb2tlPSJoc2woMjE3IDkxJSA0MCUgLyAwLjA1KSIgc3Ryb2tlLXdpZHRoPSIyIi8+PC9nPjwvc3ZnPg==')] opacity-30" />
        <div className="relative container mx-auto px-6 py-12">
          <div className="flex items-center justify-center mb-6">
            <div className="relative">
              <div className="absolute inset-0 bg-primary/20 blur-xl animate-pulse-glow rounded-full" />
              <Shield className="relative h-12 w-12 text-primary" />
            </div>
          </div>
          <h1 className="text-4xl font-bold text-center mb-3 bg-clip-text text-transparent bg-gradient-primary">
            Face Authentication System
          </h1>
          <p className="text-center text-muted-foreground max-w-2xl mx-auto">
            Advanced biometric verification with multi-model embeddings, forensic analysis, and real-time duplicate detection
          </p>
        </div>
      </section>

      {/* Stats Dashboard */}
      <section className="container mx-auto px-6 -mt-6 mb-8 relative z-10 flex-shrink-0">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <StatsCard
            icon={Database}
            label="Database Size"
            value={loading ? "..." : stats?.database_size || 0}
            description="Total faces enrolled"
            trend="up"
          />
          <StatsCard
            icon={AlertCircle}
            label="Review Queue"
            value={loading ? "..." : stats?.review_queue || 0}
            description="Pending reviews"
            variant="warning"
          />
          <StatsCard
            icon={stats?.status === "operational" ? CheckCircle : Clock}
            label="System Status"
            value={loading ? "..." : stats?.status || "unknown"}
            description="Current operational state"
            variant={stats?.status === "operational" ? "success" : "default"}
          />
        </div>
      </section>

      {/* Action Buttons */}
      <section className="container mx-auto px-6 pb-8 flex-1 flex items-center justify-center">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8 w-full max-w-5xl">
          {/* Single Verify Button */}
          <div className="relative group">
            <div className="absolute inset-0 bg-gradient-primary opacity-0 group-hover:opacity-100 blur-xl transition-opacity duration-500 rounded-2xl" />
            <Button
              onClick={() => setVerifyOpen(true)}
              size="lg"
              className="relative text-black h-40 w-full flex flex-col gap-4 hover-scale bg-card hover:bg-card/80 border-2 border-primary/20 hover:border-primary/50 shadow-elegant transition-all duration-300"
            >
              <div className="relative">
                <div className="absolute inset-0 bg-primary/20 blur-md rounded-full" />
                <Scan className="relative h-12 w-12 text-primary" />
              </div>
              <div className="text-center">
                <div className="font-semibold text-lg ">Single Verify</div>
                <div className="text-xs opacity-80">Verify one face image</div>
              </div>
            </Button>
          </div>

          {/* Batch Upload Button */}
          <div className="relative group">
            <div className="absolute inset-0 text-black bg-gradient-secondary opacity-0 group-hover:opacity-100 blur-xl transition-opacity duration-500 rounded-2xl" />
            <Button
              onClick={() => setBatchOpen(true)}
              size="lg"
              className="relative h-40 text-black w-full flex flex-col gap-4 hover-scale bg-card hover:bg-card/80 border-2 border-accent/20 hover:border-accent/50 shadow-elegant transition-all duration-300"
            >
              <div className="relative">
                <div className="absolute inset-0 bg-accent/20 blur-md rounded-full" />
                <Upload className="relative h-12 w-12 text-accent" />
              </div>
              <div className="text-center">
                <div className="font-semibold text-lg">Batch Upload</div>
                <div className="text-xs opacity-80">Process multiple images</div>
              </div>
            </Button>
          </div>

          {/* Review Queue Button */}
          <div className="relative group">
            <div className="absolute inset-0 bg-gradient-tertiary opacity-0 group-hover:opacity-100 blur-xl transition-opacity duration-500 rounded-2xl" />
            <Button
              onClick={() => setReviewOpen(true)}
              size="lg"
              className="relative h-40 w-full flex text-black flex-col gap-4 hover-scale bg-card hover:bg-card/80 border-2 border-warning/20 hover:border-warning/50 shadow-elegant transition-all duration-300"
            >
              <div className="relative">
                <div className="absolute inset-0 bg-warning/20 blur-md rounded-full" />
                <ClipboardList className="relative h-12 w-12 text-warning" />
              </div>
              <div className="text-center">
                <div className="font-semibold text-lg">Review Queue</div>
                <div className="text-xs opacity-80">Manual review pending items</div>
              </div>
            </Button>
          </div>
        </div>
      </section>

      {/* Dialogs */}
      <Dialog open={verifyOpen} onOpenChange={setVerifyOpen}>
        <DialogContent className="max-w-6xl max-h-[90vh] overflow-y-auto">
          <DialogHeader>
            <DialogTitle>Single Face Verification</DialogTitle>
          </DialogHeader>
          <VerifySection onComplete={fetchStats} />
        </DialogContent>
      </Dialog>

      <Dialog open={batchOpen} onOpenChange={setBatchOpen}>
        <DialogContent className="max-w-6xl max-h-[90vh] overflow-y-auto">
          <DialogHeader>
            <DialogTitle>Batch Upload & Deduplicate</DialogTitle>
          </DialogHeader>
          <BatchSection onComplete={fetchStats} />
        </DialogContent>
      </Dialog>

      <Dialog open={reviewOpen} onOpenChange={setReviewOpen}>
        <DialogContent className="max-w-7xl max-h-[90vh] overflow-y-auto">
          <DialogHeader>
            <DialogTitle>Review Queue</DialogTitle>
          </DialogHeader>
          <ReviewSection onUpdate={fetchStats} />
        </DialogContent>
      </Dialog>
    </div>
  );
};

export default Index;
