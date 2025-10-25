import { useState } from "react";
import { Upload } from "lucide-react";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { useToast } from "@/hooks/use-toast";
import { FaceScanLoader } from "@/components/FaceScanLoader";
import { ResultDisplay } from "@/components/ResultDisplay";

const API_BASE = import.meta.env.VITE_API_BASE || "http://127.0.0.1:8000";

interface VerifySectionProps {
  onComplete?: () => void;
}

export const VerifySection = ({ onComplete }: VerifySectionProps) => {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [applicantId, setApplicantId] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<any>(null);
  const { toast } = useToast();

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0];
    if (selectedFile) {
      setFile(selectedFile);
      setResult(null);
      const reader = new FileReader();
      reader.onload = () => setPreview(reader.result as string);
      reader.readAsDataURL(selectedFile);
    }
  };

  const handleVerify = async () => {
    if (!file) return;

    setLoading(true);
    setResult(null);

    try {
      const formData = new FormData();
      formData.append("file", file);
      if (applicantId) formData.append("applicant_id", applicantId);

      const res = await fetch(`${API_BASE}/api/verify`, {
        method: "POST",
        body: formData,
      });

      if (!res.ok) throw new Error(`Verification failed: ${res.status}`);
      
      const data = await res.json();
      setResult(data);
      
      toast({
        title: "Verification Complete",
        description: `Decision: ${data.decision}`,
      });

      onComplete?.();
    } catch (error) {
      toast({
        title: "Error",
        description: error instanceof Error ? error.message : "Failed to verify image",
        variant: "destructive",
      });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      <Card className="p-6">
        <h3 className="text-xl font-semibold mb-4">Upload Face Image</h3>
        
        <div className="grid md:grid-cols-2 gap-6">
          <div className="space-y-4">
            <div>
              <Label htmlFor="file-upload">Face Image</Label>
              <div className="mt-2">
                <Input
                  id="file-upload"
                  type="file"
                  accept="image/*"
                  onChange={handleFileChange}
                  className="cursor-pointer"
                />
              </div>
            </div>

            <div>
              <Label htmlFor="applicant-id">Applicant ID (Optional)</Label>
              <Input
                id="applicant-id"
                type="text"
                placeholder="e.g., USER_12345"
                value={applicantId}
                onChange={(e) => setApplicantId(e.target.value)}
              />
            </div>

            <Button
              onClick={handleVerify}
              disabled={!file || loading}
              className="w-full"
              size="lg"
            >
              <Upload className="mr-2 h-4 w-4" />
              {loading ? "Processing..." : "Verify Face"}
            </Button>
          </div>

          <div>
            {preview && (
              <div className="relative rounded-lg overflow-hidden border-2 border-primary/20 aspect-square">
                <img
                  src={preview}
                  alt="Preview"
                  className="w-full h-full object-cover"
                />
              </div>
            )}
          </div>
        </div>
      </Card>

      {loading && <FaceScanLoader />}

      {result && (
        <div className="mt-6">
          <h3 className="text-xl font-semibold mb-4">Verification Results</h3>
          <ResultDisplay result={result} />
        </div>
      )}
    </div>
  );
};
