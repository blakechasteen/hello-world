'use client';

import { useRef, useEffect } from 'react';
import type { ReasoningMode } from '../../app/chat/page';

interface ChatInputProps {
  value: string;
  onChange: (value: string) => void;
  onSend: () => void;
  isLoading: boolean;
  reasoningMode: ReasoningMode;
}

export function ChatInput({
  value,
  onChange,
  onSend,
  isLoading,
  reasoningMode,
}: ChatInputProps) {
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 200)}px`;
    }
  }, [value]);

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      onSend();
    }
  };

  const placeholders: Record<ReasoningMode, string> = {
    direct: 'Ask a question...',
    verify: 'Enter a claim to verify...',
    research: 'What would you like to research?',
    plan_execute: 'Describe a task to plan and execute...',
  };

  return (
    <div className="relative">
      <div className="flex items-end gap-2 bg-bg-secondary rounded-2xl border border-border-primary focus-within:border-cosmic-nebula transition-colors">
        {/* Input Area */}
        <div className="flex-1 p-3">
          <textarea
            ref={textareaRef}
            value={value}
            onChange={(e) => onChange(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={placeholders[reasoningMode]}
            disabled={isLoading}
            rows={1}
            className="w-full bg-transparent text-fg-primary placeholder-fg-tertiary resize-none focus:outline-none text-sm leading-relaxed"
            style={{ minHeight: '24px', maxHeight: '200px' }}
          />
        </div>

        {/* Action Buttons */}
        <div className="flex items-center gap-1 p-2">
          {/* Attachment Button */}
          <button
            type="button"
            className="p-2 text-fg-tertiary hover:text-fg-secondary transition-colors rounded-lg hover:bg-bg-tertiary"
            title="Attach file"
          >
            <AttachmentIcon />
          </button>

          {/* Send Button */}
          <button
            type="button"
            onClick={onSend}
            disabled={!value.trim() || isLoading}
            className={`p-2 rounded-lg transition-all ${
              value.trim() && !isLoading
                ? 'bg-cosmic-nebula text-white hover:bg-cosmic-nebula/90'
                : 'bg-bg-tertiary text-fg-tertiary cursor-not-allowed'
            }`}
            title="Send message"
          >
            {isLoading ? <LoadingSpinner /> : <SendIcon />}
          </button>
        </div>
      </div>

      {/* Quick Actions */}
      <div className="absolute -top-10 left-0 flex items-center gap-2">
        <QuickAction icon={<ExampleIcon />} label="Examples" />
        <QuickAction icon={<HistoryIcon />} label="History" />
        <QuickAction icon={<BookmarkIcon />} label="Saved" />
      </div>
    </div>
  );
}

function QuickAction({ icon, label }: { icon: React.ReactNode; label: string }) {
  return (
    <button className="flex items-center gap-1.5 px-2 py-1 text-xs text-fg-tertiary hover:text-fg-secondary hover:bg-bg-secondary rounded transition-colors">
      {icon}
      {label}
    </button>
  );
}

function AttachmentIcon() {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="m21.44 11.05-9.19 9.19a6 6 0 0 1-8.49-8.49l8.57-8.57A4 4 0 1 1 18 8.84l-8.59 8.57a2 2 0 0 1-2.83-2.83l8.49-8.48" />
    </svg>
  );
}

function SendIcon() {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <line x1="22" y1="2" x2="11" y2="13" />
      <polygon points="22 2 15 22 11 13 2 9 22 2" />
    </svg>
  );
}

function LoadingSpinner() {
  return (
    <svg className="animate-spin" xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <circle cx="12" cy="12" r="10" strokeOpacity="0.25" />
      <path d="M12 2a10 10 0 0 1 10 10" strokeOpacity="1" />
    </svg>
  );
}

function ExampleIcon() {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2" />
    </svg>
  );
}

function HistoryIcon() {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="12" cy="12" r="10" />
      <polyline points="12 6 12 12 16 14" />
    </svg>
  );
}

function BookmarkIcon() {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="m19 21-7-5-7 5V5a2 2 0 0 1 2-2h10a2 2 0 0 1 2 2z" />
    </svg>
  );
}
