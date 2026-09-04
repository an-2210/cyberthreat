import React, { useState } from 'react';
import { Upload, FileText, CheckCircle2, AlertCircle, RefreshCw } from 'lucide-react';
import { datasetApi } from '../../api/datasetApi';

interface DatasetUploaderProps {
  onUploadSuccess?: () => void;
}

export const DatasetUploader: React.FC<DatasetUploaderProps> = ({ onUploadSuccess }) => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [datasetType, setDatasetType] = useState<'CIC-IDS2017' | 'UNSW-NB15' | 'Custom CSV'>('CIC-IDS2017');
  const [isUploading, setIsUploading] = useState(false);
  const [statusMessage, setStatusMessage] = useState<{ text: string; isError: boolean } | null>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setSelectedFile(e.target.files[0]);
      setStatusMessage(null);
    }
  };

  const handleUpload = async () => {
    if (!selectedFile) {
      setStatusMessage({ text: 'Please select a CSV dataset file first.', isError: true });
      return;
    }

    setIsUploading(true);
    setStatusMessage(null);

    try {
      const res = await datasetApi.uploadDataset(selectedFile, datasetType);
      setStatusMessage({ text: res.message || 'Dataset uploaded successfully.', isError: false });
      if (onUploadSuccess) onUploadSuccess();
    } catch (err: any) {
      setStatusMessage({
        text: err?.response?.data?.message || 'Dataset upload failed or backend API unavailable.',
        isError: true
      });
    } finally {
      setIsUploading(false);
    }
  };

  return (
    <div className="soc-card border border-soc-border p-5 space-y-4">
      <div className="flex items-center justify-between border-b border-soc-border/60 pb-3">
        <h3 className="text-xs font-mono font-bold uppercase tracking-wider text-slate-200 flex items-center space-x-2">
          <Upload className="w-4 h-4 text-cyan-400" />
          <span>Ingest & Analyze Dataset (Phase 1 EDA)</span>
        </h3>
        <div className="text-[10px] font-mono text-slate-400">
          Supported Format: <span className="text-slate-200 font-semibold">.CSV</span>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 items-center">
        {/* Dataset Adapter Selector */}
        <div>
          <label className="block text-[11px] font-mono text-slate-400 mb-1.5 uppercase">
            Dataset Adapter Preset
          </label>
          <select
            value={datasetType}
            onChange={(e) => setDatasetType(e.target.value as any)}
            className="w-full bg-slate-900 border border-soc-border rounded-md px-3 py-2 text-xs font-mono text-slate-200 focus:outline-none focus:border-cyan-500/60"
          >
            <option value="CIC-IDS2017">UNB CIC-IDS2017 (84 Features)</option>
            <option value="UNSW-NB15">UNSW Canberra NB15 (49 Features)</option>
            <option value="Custom CSV">Generic Network CSV</option>
          </select>
        </div>

        {/* Dropzone File Picker */}
        <div className="md:col-span-2">
          <label className="block text-[11px] font-mono text-slate-400 mb-1.5 uppercase">
            Select CSV File
          </label>
          <div className="flex items-center space-x-3">
            <label className="flex-1 flex items-center justify-between px-3 py-2 rounded-md bg-slate-900 border border-dashed border-soc-border cursor-pointer hover:border-cyan-500/60 transition-colors">
              <div className="flex items-center space-x-2 truncate">
                <FileText className="w-4 h-4 text-cyan-400 shrink-0" />
                <span className="text-xs font-mono text-slate-300 truncate">
                  {selectedFile ? selectedFile.name : 'Choose CSV dataset or drag file here...'}
                </span>
              </div>
              <input
                type="file"
                accept=".csv"
                onChange={handleFileChange}
                className="hidden"
              />
            </label>

            <button
              onClick={handleUpload}
              disabled={isUploading || !selectedFile}
              className="px-4 py-2 rounded-md text-xs font-mono font-bold uppercase bg-cyan-950/90 text-cyan-300 border border-cyan-800/80 hover:bg-cyan-900 transition-colors disabled:opacity-40 disabled:cursor-not-allowed flex items-center space-x-2 shrink-0"
            >
              {isUploading ? (
                <>
                  <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                  <span>Analyzing...</span>
                </>
              ) : (
                <>
                  <Upload className="w-3.5 h-3.5" />
                  <span>Run Phase 1 Audit</span>
                </>
              )}
            </button>
          </div>
        </div>
      </div>

      {/* Status Alert Banner */}
      {statusMessage && (
        <div
          className={`p-3 rounded-md border text-xs font-mono flex items-center space-x-2 ${
            statusMessage.isError
              ? 'bg-rose-950/40 text-rose-300 border-rose-900/60'
              : 'bg-emerald-950/40 text-emerald-300 border-emerald-900/60'
          }`}
        >
          {statusMessage.isError ? (
            <AlertCircle className="w-4 h-4 shrink-0 text-rose-400" />
          ) : (
            <CheckCircle2 className="w-4 h-4 shrink-0 text-emerald-400" />
          )}
          <span>{statusMessage.text}</span>
        </div>
      )}
    </div>
  );
};
