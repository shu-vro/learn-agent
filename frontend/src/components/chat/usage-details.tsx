"use client";

import { useState } from "react";
import { Bar, BarChart, CartesianGrid, XAxis } from "recharts";

import { MessageAction } from "@/components/ai-elements/message";
import {
  type ChartConfig,
  ChartContainer,
  ChartLegend,
  ChartLegendContent,
  ChartTooltip,
  ChartTooltipContent,
} from "@/components/ui/chart";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import type { ChatUsage } from "@/lib/api/chat";

const chartConfig = {
  input: {
    label: "Input",
    color: "var(--chart-1)",
  },
  cache: {
    label: "Cache",
    color: "var(--chart-2)",
  },
  output: {
    label: "Output",
    color: "var(--chart-3)",
  },
} satisfies ChartConfig;

function formatTokens(n: number): string {
  return new Intl.NumberFormat("en-US").format(n);
}

export function UsageDetailsButton({ usage }: { usage: ChatUsage }) {
  const [open, setOpen] = useState(false);

  const chartData =
    usage.iteration_details.length > 0
      ? usage.iteration_details.map((item) => ({
          iteration: `I${item.iteration}`,
          input: item.input_token,
          cache: item.cache_token,
          output: item.output_token,
        }))
      : [
          {
            iteration: "Total",
            input: usage.input_token,
            cache: usage.cache_token,
            output: usage.output_token,
          },
        ];

  return (
    <>
      <MessageAction
        label="Details"
        tooltip="Usage details"
        size="sm"
        onClick={() => setOpen(true)}
      >
        Details
      </MessageAction>
      <Dialog open={open} onOpenChange={setOpen}>
        <DialogContent className="sm:max-w-lg">
          <DialogHeader>
            <DialogTitle>Usage details</DialogTitle>
            <DialogDescription>
              {usage.iterations} iteration
              {usage.iterations === 1 ? "" : "s"}
            </DialogDescription>
          </DialogHeader>

          <div className="grid grid-cols-3 gap-3">
            <div className="flex flex-col gap-1 rounded-xl bg-muted/50 p-3">
              <span className="text-muted-foreground text-xs">Input</span>
              <span className="font-medium tabular-nums">
                {formatTokens(usage.input_token)}
              </span>
            </div>
            <div className="flex flex-col gap-1 rounded-xl bg-muted/50 p-3">
              <span className="text-muted-foreground text-xs">Cache</span>
              <span className="font-medium tabular-nums">
                {formatTokens(usage.cache_token)}
              </span>
            </div>
            <div className="flex flex-col gap-1 rounded-xl bg-muted/50 p-3">
              <span className="text-muted-foreground text-xs">Output</span>
              <span className="font-medium tabular-nums">
                {formatTokens(usage.output_token)}
              </span>
            </div>
          </div>

          <ChartContainer config={chartConfig} className="min-h-50 w-full">
            <BarChart accessibilityLayer data={chartData}>
              <CartesianGrid vertical={false} />
              <XAxis
                dataKey="iteration"
                tickLine={false}
                tickMargin={8}
                axisLine={false}
              />
              <ChartTooltip content={<ChartTooltipContent />} />
              <ChartLegend content={<ChartLegendContent />} />
              <Bar dataKey="input" fill="var(--color-input)" radius={3} />
              <Bar dataKey="cache" fill="var(--color-cache)" radius={3} />
              <Bar dataKey="output" fill="var(--color-output)" radius={3} />
            </BarChart>
          </ChartContainer>
        </DialogContent>
      </Dialog>
    </>
  );
}
