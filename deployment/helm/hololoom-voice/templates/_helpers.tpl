{{/*
Expand the name of the chart.
*/}}
{{- define "hololoom-voice.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
*/}}
{{- define "hololoom-voice.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "hololoom-voice.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "hololoom-voice.labels" -}}
helm.sh/chart: {{ include "hololoom-voice.chart" . }}
{{ include "hololoom-voice.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "hololoom-voice.selectorLabels" -}}
app.kubernetes.io/name: {{ include "hololoom-voice.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
app: voice-agent
component: backend
{{- end }}

{{/*
Create the name of the service account to use
*/}}
{{- define "hololoom-voice.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (include "hololoom-voice.fullname" .) .Values.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.name }}
{{- end }}
{{- end }}
