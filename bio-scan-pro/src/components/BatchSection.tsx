import { useState, useCallback } from "react";
import { Upload, X } from "lucide-react";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { useToast } from "@/hooks/use-toast";
import { FaceScanLoader } from "@/components/FaceScanLoader";
import { BatchResultCard } from "@/components/BatchResultCard";

const API_BASE = import.meta.env.VITE_API_BASE || "http://127.0.0.1:8000";

interface BatchSectionProps {
  onComplete?: () => void;
}

export const BatchSection = ({ onComplete }: BatchSectionProps) => {
  const [files, setFiles] = useState<File[]>([]);
  const [previews, setPreviews] = useState<{ name: string; src: string }[]>([]);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState<any>(null);
  const [dragActive, setDragActive] = useState(false);
  const { toast } = useToast();

  const handleFiles = useCallback((newFiles: File[]) => {
    setFiles(newFiles);
    setResults(null);

    const readers = newFiles.map((file) => {
      return new Promise<{ name: string; src: string }>((resolve) => {
        const reader = new FileReader();
        reader.onload = () => resolve({ name: file.name, src: reader.result as string });
        reader.readAsDataURL(file);
      });
    });

    Promise.all(readers).then(setPreviews);
  }, []);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragActive(false);
      const droppedFiles = Array.from(e.dataTransfer.files);
      handleFiles(droppedFiles);
    },
    [handleFiles]
  );

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      handleFiles(Array.from(e.target.files));
    }
  };

  const handleSubmit = async () => {
    if (!files.length) return;

    setLoading(true);
    setResults(null);

    try {
      const formData = new FormData();
      files.forEach((file) => formData.append("files", file));

      const res = await fetch(`${API_BASE}/api/batch/deduplicate`, {
        method: "POST",
        body: formData,
      });

      if (!res.ok) throw new Error(`Batch processing failed: ${res.status}`);
      
      const data = await res.json();
      setResults(data);
      
      toast({
        title: "Batch Processing Complete",
        description: `Processed ${data.total_processed} images`,
      });

      onComplete?.();
    } catch (error) {
      toast({
        title: "Error",
        description: error instanceof Error ? error.message : "Failed to process batch",
        variant: "destructive",
      });
    } finally {
      setLoading(false);
    }
  };

  const clearFiles = () => {
    setFiles([]);
    setPreviews([]);
    setResults(null);
  };

  return (
    <div className="space-y-6">
      <Card className="p-6">
        <h3 className="text-xl font-semibold mb-4">Batch Upload</h3>

        <div
          onDragOver={(e) => {
            e.preventDefault();
            setDragActive(true);
          }}
          onDragLeave={() => setDragActive(false)}
          onDrop={handleDrop}
          onClick={() => document.getElementById("batch-input")?.click()}
          className={`border-2 border-dashed rounded-xl p-12 text-center cursor-pointer transition-all duration-300 ${
            dragActive
              ? "border-primary bg-primary/5 scale-105"
              : "border-muted hover:border-primary/50 hover:bg-muted/50"
          }`}
        >
          <Upload className="h-12 w-12 mx-auto mb-4 text-muted-foreground" />
          <p className="text-lg font-medium mb-2">
            Drag & drop images here, or click to browse
          </p>
          <p className="text-sm text-muted-foreground">
            Upload multiple face images for batch deduplication
          </p>
          <input
            id="batch-input"
            type="file"
            accept="image/*"
            multiple
            className="hidden"
            onChange={handleChange}
          />
        </div>

        {previews.length > 0 && (
          <div className="mt-6">
            <div className="flex items-center justify-between mb-4">
              <p className="text-sm text-muted-foreground">
                {previews.length} file(s) selected
              </p>
              <Button variant="ghost" size="sm" onClick={clearFiles}>
                <X className="h-4 w-4 mr-2" />
                Clear All
              </Button>
            </div>

            <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6 gap-4 mb-6">
              {previews.map((preview, index) => (
                <div key={index} className="relative group">
                  <div className="aspect-square rounded-lg overflow-hidden border-2 border-primary/20">
                    <img
                      src={preview.src}
                      alt={preview.name}
                      className="w-full h-full object-cover"
                    />
                  </div>
                  <p className="text-xs text-center mt-1 truncate">{preview.name}</p>
                </div>
              ))}
            </div>

            <Button onClick={handleSubmit} disabled={loading} size="lg" className="w-full">
              <Upload className="mr-2 h-4 w-4" />
              {loading ? "Processing..." : "Process Batch"}
            </Button>
          </div>
        )}
      </Card>

      {loading && <FaceScanLoader />}

      {results?.results && (
        <div className="space-y-4">
          <h3 className="text-2xl font-semibold">Results</h3>
          <div className="grid gap-6">
            {Object.entries(results.results).map(([filename, result]: [string, any]) => (
              <BatchResultCard key={filename} filename={filename} result={result} />
            ))}
          </div>
        </div>
      )}
    </div>
  );
};
