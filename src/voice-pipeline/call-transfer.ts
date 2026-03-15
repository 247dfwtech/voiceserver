import twilio from "twilio";

// Lazy-initialized Twilio client
let _client: ReturnType<typeof twilio> | null = null;

function getTwilioClient(): ReturnType<typeof twilio> {
  if (!_client) {
    const sid = process.env.TWILIO_ACCOUNT_SID;
    const token = process.env.TWILIO_AUTH_TOKEN;
    if (!sid || !token) throw new Error("TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN required for call transfer");
    _client = twilio(sid, token);
  }
  return _client;
}

interface TransferOptions {
  callSid: string;
  destination: string;
  publicUrl: string;
  callerNumber: string;
  timeout?: number;
}

/** XML-escape a string to prevent TwiML injection */
function escapeXml(str: string): string {
  return str
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&apos;");
}

/** Validate a phone number (E.164 or SIP URI) */
function isValidDestination(dest: string): boolean {
  // E.164: +1234567890 (7-15 digits)
  if (/^\+[1-9]\d{6,14}$/.test(dest)) return true;
  // SIP URI
  if (/^sip:[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+$/.test(dest)) return true;
  return false;
}

/** Validate DTMF digits (0-9, *, #, w for wait) */
function isValidDTMF(digits: string): boolean {
  return /^[0-9*#wW]+$/.test(digits);
}

/**
 * Transfer an active call using Twilio's <Dial> verb.
 * The original leg hears hold music while the transfer connects.
 * If the transfer fails, the caller is reconnected to the media stream.
 */
export async function transferCallWithDial(opts: TransferOptions): Promise<{ success: boolean; error?: string }> {
  const client = getTwilioClient();

  // Validate destination to prevent TwiML injection
  if (!isValidDestination(opts.destination)) {
    return { success: false, error: `Invalid destination format: ${opts.destination.slice(0, 20)}` };
  }

  const timeout = Math.min(Math.max(opts.timeout || 30, 5), 120);

  try {
    await client.calls(opts.callSid).update({
      twiml: `<Response>
        <Dial callerId="${escapeXml(opts.callerNumber)}" timeout="${timeout}" action="${escapeXml(opts.publicUrl)}/api/webhooks/twilio/transfer-status">
          <Number>${escapeXml(opts.destination)}</Number>
        </Dial>
      </Response>`,
    });

    return { success: true };
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    console.error(`[call-transfer] Dial transfer failed:`, message);
    return { success: false, error: message };
  }
}

/**
 * Transfer using conference bridge approach.
 */
export async function transferCallWithConference(opts: TransferOptions): Promise<{ success: boolean; conferenceSid?: string; error?: string }> {
  const client = getTwilioClient();

  if (!isValidDestination(opts.destination)) {
    return { success: false, error: `Invalid destination format` };
  }

  // Conference name is server-generated -- safe from injection
  const conferenceName = `transfer-${opts.callSid}-${Date.now()}`;

  try {
    await client.calls(opts.callSid).update({
      twiml: `<Response>
        <Dial>
          <Conference waitUrl="" startConferenceOnEnter="true" endConferenceOnExit="true">${escapeXml(conferenceName)}</Conference>
        </Dial>
      </Response>`,
    });

    const outboundLeg = await client.calls.create({
      to: opts.destination,
      from: opts.callerNumber,
      twiml: `<Response>
        <Dial>
          <Conference startConferenceOnEnter="true" endConferenceOnExit="false">${escapeXml(conferenceName)}</Conference>
        </Dial>
      </Response>`,
    });

    return { success: true, conferenceSid: outboundLeg.sid };
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    console.error(`[call-transfer] Conference transfer failed:`, message);
    return { success: false, error: message };
  }
}

/**
 * Send DTMF tones on an active call (for IVR navigation).
 */
export async function sendDTMF(callSid: string, digits: string): Promise<void> {
  if (!isValidDTMF(digits)) {
    throw new Error(`Invalid DTMF digits: only 0-9, *, #, w allowed`);
  }

  const client = getTwilioClient();
  await client.calls(callSid).update({
    twiml: `<Response><Play digits="${escapeXml(digits)}"/></Response>`,
  });
}
